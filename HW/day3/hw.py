from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import StateGraph, END, add_messages

import time
import requests


# ==================== ASR 設定 ====================
BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
auth = ("nutc2504", "nutc2504")

# ==================== LLM 設定 ====================
llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=0
)

# ==================== Tools 定義 ====================

@tool
def asr_transcribe(audio_path: str):
    """
    使用 ASR 工具將語音檔案轉換為文字。
    輸入參數 audio_path 是音檔的路徑。
    返回包含 TXT 和 SRT 內容的字典。
    """
    try:
        # 建立任務
        with open(audio_path, "rb") as f:
            r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
        r.raise_for_status()
        task_id = r.json()["id"]
        
        print(f"ASR Task ID: {task_id}")
        print("等待轉文字...")
        
        txt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT"
        srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"
        
        def wait_download(url: str, max_tries=600):
            for _ in range(max_tries):
                try:
                    resp = requests.get(url, timeout=(5, 60), auth=auth)
                    if resp.status_code == 200:
                        return resp.text
                except requests.exceptions.ReadTimeout:
                    pass
                time.sleep(2)
            return None
        
        # 等待 TXT 和 SRT
        txt_text = wait_download(txt_url, max_tries=600)
        srt_text = wait_download(srt_url, max_tries=600)
        
        if txt_text is None:
            return {"error": "ASR 轉錄逾時或錯誤"}
    
        return {
            "txt": txt_text,
            "srt": srt_text
        }
    except Exception as e:
        return {"error": f"ASR 處理失敗: {str(e)}"}


@tool
def extract_minutes(srt_content: str):
    """
    從 SRT 格式內容提取逐字稿。
    將時間軸與對應台詞整理成清晰格式。
    """
    if not srt_content or srt_content == "SRT 內容不可用":
        return "無法提取逐字稿：SRT 內容不可用"
    
    lines = srt_content.strip().split('\n')
    formatted_transcript = []
    
    i = 0
    while i < len(lines):
        # SRT 格式: 序號 / 時間軸 / 台詞 / 空行
        if lines[i].strip().isdigit():
            seq_num = lines[i].strip()
            if i + 1 < len(lines):
                timestamp = lines[i + 1].strip()
                dialogue = []
                i += 2
                while i < len(lines) and lines[i].strip():
                    dialogue.append(lines[i].strip())
                    i += 1
                
                formatted_transcript.append(
                    f"[{timestamp}]\n{' '.join(dialogue)}"
                )
        i += 1
    
    return "\n\n".join(formatted_transcript)


@tool  
def generate_summary(full_text: str):
    """
    根據完整文字內容生成重點摘要。
    提取關鍵信息、主要論點和重要結論。
    """
    # 使用 LLM 生成摘要
    summary_prompt = f"""請根據以下內容生成重點摘要。摘要應該：
1. 提取關鍵主題和重要信息
2. 保持簡潔，約 200-300 字
3. 使用條列式呈現主要重點
4. 用繁體中文撰寫

內容：
{full_text}

請生成摘要："""
    
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return response.content


@tool
def save_results(task_id: str, minutes: str, summary: str):
    """
    將逐字稿和摘要儲存到檔案。
    """
    from pathlib import Path
    
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    
    # 儲存逐字稿
    minutes_path = out_dir / f"{task_id}_minutes.txt"
    minutes_path.write_text(minutes, encoding="utf-8")
    
    # 儲存摘要
    summary_path = out_dir / f"{task_id}_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    
    return f"結果已儲存:\n逐字稿: {minutes_path}\n摘要: {summary_path}"


# ==================== State 定義 ====================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    audio_path: str
    asr_result: dict
    minutes: str
    summary: str
    final_output: str


# ==================== Nodes 定義 ====================

def asr_node(state: AgentState):
    """ASR 節點：執行語音轉文字"""
    print("\n=== ASR Node: 開始語音轉文字 ===")
    audio_path = state.get("audio_path", "./audio/Podcast_EP14_30s.wav")
    
    # 執行 ASR
    result = asr_transcribe.invoke({"audio_path": audio_path})
    
    return {
        "asr_result": result,
        "messages": [HumanMessage(content=f"ASR 轉錄完成，Task ID: {result.get('task_id', 'N/A')}")]
    }


def minutes_taker_node(state: AgentState):
    """逐字稿節點：整理詳細逐字稿"""
    print("\n=== Minutes Taker Node: 生成逐字稿 ===")
    asr_result = state["asr_result"]
    
    if "error" in asr_result:
        minutes = f"錯誤: {asr_result['error']}"
    else:
        srt_content = asr_result.get("srt", "")
        minutes = extract_minutes.invoke({"srt_content": srt_content})
    
    return {
        "minutes": minutes,
        "messages": [HumanMessage(content="逐字稿生成完成")]
    }


def summarizer_node(state: AgentState):
    """摘要節點：生成重點摘要"""
    print("\n=== Summarizer Node: 生成重點摘要 ===")
    asr_result = state["asr_result"]
    
    if "error" in asr_result:
        summary = f"錯誤: {asr_result['error']}"
    else:
        full_text = asr_result.get("txt", "")
        summary = generate_summary.invoke({"full_text": full_text})
    
    return {
        "summary": summary,
        "messages": [HumanMessage(content="摘要生成完成")]
    }


def writer_node(state: AgentState):
    """寫入節點：儲存結果"""
    print("\n=== Writer Node: 儲存結果 ===")
    task_id = state["asr_result"].get("task_id", "unknown")
    minutes = state.get("minutes", "")
    summary = state.get("summary", "")
    
    save_result = save_results.invoke({
        "task_id": task_id,
        "minutes": minutes,
        "summary": summary
    })
    
    return {
        "final_output": save_result,
        "messages": [HumanMessage(content=save_result)]
    }


# ==================== Graph 建構 ====================

workflow = StateGraph(AgentState)

# 加入節點
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

# 設定入口
workflow.set_entry_point("asr")

# 加入邊
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)

# 編譯
app = workflow.compile()

# ==================== 執行 ====================

if __name__ == "__main__":
    print("=== 語音轉文字處理系統 (測試版) ===")
    print(app.get_graph().draw_ascii())
    print("\n")
    
    # 設定音檔路徑
    audio_file = input("請輸入音檔路徑 (直接按 Enter 使用預設: ./audio/Podcast_EP14_30s.wav): ").strip()
    if not audio_file:
        audio_file = "./audio/Podcast_EP14_30s.wav"

    # 初始化狀態
    initial_state = {
        "messages": [],
        "audio_path": audio_file,
        "asr_result": {},
        "minutes": "",
        "summary": "",
        "final_output": ""
    }
    
    # 執行工作流
    print("開始處理...")
    print("=" * 60)
    
    final_state = app.invoke(initial_state)
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("處理完成！")
    print("=" * 60)
    print(f"\n{final_state['final_output']}")
    
    # 顯示預覽
    print("\n--- 逐字稿預覽 ---")
    print(final_state['minutes'])
    
    print("\n--- 摘要預覽 ---")
    print(final_state['summary'])
