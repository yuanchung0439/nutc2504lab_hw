import json, os, requests, base64
from typing import Annotated, TypedDict, Literal, Dict, List, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from playwright.sync_api import sync_playwright
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

SEARXNG_URL = "https://puli-8080.huannago.com/search"

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

CACHE_FILE = "query_cache.json"

def load_cache() -> Dict:
    """載入快取"""
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}


def save_cache(cache: Dict):
    """儲存快取"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_cache_key(query: str) -> str:
    """生成快取鍵值"""
    return query.strip().lower()


class AgentState(TypedDict):
    """AI Agent 的狀態"""
    input: str  # 使用者輸入的問題
    knowledge_base: List[Dict]  # 已蒐集的知識庫
    search_queries: List[str]  # 生成的搜尋關鍵字
    search_results: List[Dict]  # 搜尋結果
    vlm_results: List[Dict]
    final_answer: Optional[str]  # 最終答案
    decision: str  # 決策結果 ('continue' 或 'finish')
    iteration: int  # 當前迭代次數
    max_iterations: int  # 最大迭代次數
    round_number: int # Round 計數


def check_cache_node(state: AgentState):
    """檢查快取中是否有相同問題的答案"""
    print("\n" + "="*50)
    print(f"🚀 開始處理問題: {state['input']}")
    print(f"🔍 [Node] 檢查快取: {state['input']}")
    
    cache = load_cache()
    cache_key = get_cache_key(state['input'])
    
    if cache_key in cache:
        print("✅ 快取命中！直接返回先前的答案")
        state['final_answer'] = cache[cache_key]['answer']
        state['decision'] = 'finish'
        state['knowledge_base'] = cache[cache_key].get('knowledge_base', [])
    else:
        print("❌ 未命中快取，進入 Agent 思考流程。")
        state['decision'] = 'continue'
    
    return state


def planner_node(state: AgentState):
    """規劃查詢策略，判斷是否需要更多資訊"""
    print("\n" + "="*50)
    print(f"✨ [Think] Round {state['round_number']}")
    print("🧠 [Node] Planner - 評估當前知識是否足夠...")
    
    # 檢查迭代次數
    if state['iteration'] >= state['max_iterations']:
        print(f"⚠️ 已達最大迭代次數 ({state['max_iterations']})，強制結束")
        state['decision'] = 'finish'
        return state
    
    # 構建評估提示
    prompt = f"""你是一個資訊評估專家。請評估以下情況：

    使用者問題：{state['input']}

    目前已蒐集的資訊：
    {json.dumps(state['knowledge_base'], ensure_ascii=False, indent=2) if state['knowledge_base'] else '目前沒有任何資訊'}

    任務：判斷目前的資訊是否足以回答使用者的問題。

    回答格式（只需回答 YES 或 NO）：
    - YES：如果資訊充足且可以給出準確答案
    - NO：如果需要更多資訊才能回答

    你的判斷："""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        decision_text = response.content.strip().upper()
        
        if "YES" in decision_text:
            print("✅ 判斷：資訊充足，可以生成最終答案")
            state['decision'] = 'finish'
        else:
            print("❌ 判斷：資訊不足，需要繼續搜尋")
            state['decision'] = 'continue'
            state['iteration'] += 1
    except Exception as e:
        print(f"⚠️ Planner 出錯：{e}")
        state['decision'] = 'finish'  # 出錯時結束流程
    
    return state

def query_gen_node(state: AgentState) -> AgentState:
    """生成搜尋關鍵字"""
    print("\n" + "="*50)
    print("🔑 [Node] Query Generator - 生成搜尋關鍵字...")
    
    prompt = f"""你是一個搜尋關鍵字生成專家。

    使用者問題：{state['input']}

    已搜尋過的關鍵字：{state['search_queries']}

    任務：根據使用者問題生成 1-2 個**新的**繁體中文搜尋關鍵字，這些關鍵字應該：
    1. 與問題高度相關
    2. 不重複先前搜尋過的關鍵字
    3. 能夠找到最新、最準確的資訊

    請直接列出關鍵字，每行一個，不要有其他說明文字。

    關鍵字："""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        keywords = [k.strip() for k in response.content.strip().split('\n') if k.strip()]
        keywords = [k for k in keywords if k not in state['search_queries']][:2]  # 最多2個新關鍵字
        
        if keywords:
            print(f"✅ 生成關鍵字：{keywords}")
            state['search_queries'].extend(keywords)
        else:
            print("⚠️ 無法生成新關鍵字")
            state['decision'] = 'finish'
    except Exception as e:
        print(f"⚠️ Query Generator 出錯：{e}")
        state['decision'] = 'finish'
    
    state['round_number'] += 1

    return state


def search_tool_node(state: AgentState) -> AgentState:
    """執行網頁搜尋"""
    print("\n" + "="*50)
    print("🌐 [Node] Search Tool - 執行搜尋...")
    
    if not state['search_queries']:
        print("⚠️ 沒有搜尋關鍵字")
        return state
    
    latest_query = state['search_queries'][-1]  # 使用最新的關鍵字
    print(f"📍 搜尋關鍵字：{latest_query}")
    
    # 呼叫 SearXNG
    params = {
        "q": latest_query,
        "format": "json",
        "language": "zh-TW"
    }

    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])[:3]  # 取前3筆
        
        if results:
            print(f"✅ 找到 {len(results)} 筆結果")
            state['search_results'] = results

            
            
            # 將結果加入知識庫
            for idx, r in enumerate(results, 1):
                state['knowledge_base'].append({
                    'source': 'search',
                    'title': r.get('title', '無標題'),
                    'url': r.get('url', ''),
                    'content': r.get('content', '無摘要')[:300]
                })
                vlm_processor(r.get('url', ''), r.get('title', '無標題'), state)
                print(f"  [{idx}] {r.get('title', '無標題')}")

        else:
            print("❌ 沒有找到搜尋結果")
    except Exception as e:
        print(f"⚠️ 搜尋出錯：{e}")
    
    return state


def vlm_processor(url: str, title: str, state: AgentState) -> AgentState:
    """
    使用 Playwright 滾動截圖，並使用多模態 LLM 讀取網頁內容。
    """
    print(f"📸 [VLM] 啟動視覺閱讀: {url}")
    
    def capture_rolling_screenshots(url, output_dir="scans_temp"):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        screenshots_b64 = []
        
        try:
            with sync_playwright() as p:
                # 啟動瀏覽器 (Headless 模式)
                browser = p.chromium.launch(
                    headless=True, 
                    args=["--disable-blink-features=AutomationControlled"] # 規避部分反爬蟲
                )
                
                # 設定 viewport (模擬桌面瀏覽)
                context = browser.new_context(viewport={'width': 1280, 'height': 1200})
                page = context.new_page()
                
                # 前往網頁
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(3000) # 等待渲染
                
                # --- CSS Injection (去廣告/彈窗) ---
                page.add_style_tag(content="""
                    iframe { opacity: 0 !important; pointer-events: none !important; }
                    div[id*='cookie'], div[class*='cookie'], div[id*='ads'], div[class*='ads'] { display: none !important; }
                    div[class*='overlay'], div[id*='overlay'], div[class*='popup'] { opacity: 0 !important; pointer-events: none !important; }
                    header, nav { position: absolute !important; } /* 防止 sticky header 遮擋截圖 */
                """)

                total_height = page.evaluate("document.body.scrollHeight")
                viewport_height = 1200
                current_scroll = 0
                
                for i in range(3):
                    # 滾動
                    page.evaluate(f"window.scrollTo(0, {current_scroll})")
                    page.wait_for_timeout(1000) # 等待滾動後渲染
                    
                    # 截圖並轉 Base64
                    b64 = base64.b64encode(page.screenshot()).decode('utf-8')
                    screenshots_b64.append(b64)
                    print(f"   - 截圖 {i+1} 完成 (Scroll: {current_scroll})")
                    
                    current_scroll += (viewport_height - 200) # 重疊 200px 避免割裂文字
                    if current_scroll >= total_height: break
                    
                browser.close()
        except Exception as e:
            print(f"❌ 截圖失敗: {e}")
        
        state["vlm_results"] = screenshots_b64
        return state

    # 執行截圖
    images = capture_rolling_screenshots(url)
    
    if not images: 
        return "錯誤：無法讀取網頁內容或截圖失敗。"

    print(f"🤖 [LLM] 正在分析 {len(images)} 張圖片...")

    # --- 組裝多模態訊息 ---
    msg_content = [
        {
            "type": "text", 
            "text": f"這是一個網頁的滾動截圖，標題為：{title}。\n請忽略廣告與導航欄，摘要此網頁的核心內容，並特別關注任何數據、日期或具體事實。"
        }
    ]
    
    # 加入所有圖片
    for img in images:
        msg_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })
    
    # 呼叫 LLM
    try:
        response = llm.invoke([HumanMessage(content=msg_content)])
        return response.content
    except Exception as e:
        return f"LLM 分析失敗: {e}"


def final_answer_node(state: AgentState) -> AgentState:
    """生成最終答案"""
    print("\n" + "="*50)
    print("📝 [Node] Final Answer - 生成最終答案...")
    
    if not state['knowledge_base']:
        state['final_answer'] = "抱歉，我無法找到足夠的資訊來回答您的問題。"
        return state
    
    # 構建答案生成提示
    knowledge_summary = "\n\n".join([
        f"來源 {idx+1}：{item['title']}\n{item['content']}"
        for idx, item in enumerate(state['knowledge_base'])
    ])
    
    prompt = f"""你是一個專業的資訊整合助手。請根據以下資料回答使用者的問題。

    使用者問題：{state['input']}

    參考資料：
    {knowledge_summary}

    要求：
    1. 根據參考資料提供準確、完整的答案
    2. 如果資料中有矛盾，請指出並說明
    3. 答案要清晰、有條理
    4. 如果資料不足，請誠實說明

    你的答案："""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        state['final_answer'] = response.content.strip()
        print("✅ 答案生成完成")
        
        # 儲存到快取
        cache = load_cache()
        cache_key = get_cache_key(state['input'])
        cache[cache_key] = {
            'answer': state['final_answer'],
            'knowledge_base': state['knowledge_base']
        }
        save_cache(cache)
        print("💾 已儲存到快取")
    except Exception as e:
        print(f"⚠️ 答案生成出錯：{e}")
        state['final_answer'] = "抱歉，生成答案時發生錯誤。"
    
    return state


def route_after_cache(state):
    return "planner" if state['decision'] == 'continue' else "final_answer"


def route_after_planner(state):
    return "query_gen" if state['decision'] == 'continue' else "final_answer"


workflow = StateGraph(AgentState)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")

workflow.add_conditional_edges(
    "check_cache",
    route_after_cache,
    {
        "planner": "planner",
        "final_answer": "final_answer"
    }
)

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "query_gen": "query_gen",
        "final_answer": "final_answer"
    }
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")  # 搜尋後回到 planner 評估
workflow.add_edge("final_answer", END)

app = workflow.compile()
print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    while True:
        user_input = input("\n請輸入問題: ")
        if user_input.lower() in ["exit", "q"]: break
        inputs = {
            "input": user_input,
            "knowledge_base": [],
            "search_queries": [],
            "search_results": [],
            "vlm_results": [],
            "final_answer": None,
            "decision": "continue",
            "iteration": 0,
            "max_iterations": 3,
            "round_number": 0
        }
        result = app.invoke(inputs)

        # 輸出結果
        print("\n" + "="*60)
        print("✨ 查證完成！")
        print("="*60)
        print(f"\n【最終答案】\n{result['final_answer']}")
        print(f"\n【共使用 {len(result['knowledge_base'])} 個資料來源】")
        print(f"【執行了 {result['iteration']} 次迭代】")