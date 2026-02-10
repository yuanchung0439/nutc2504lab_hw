import os
import pandas as pd
import requests
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI
import uuid
from typing import List, Tuple


EMBEDDING_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "day6_cw_knowledge_base"

client = QdrantClient(url="http://localhost:6333")

# LangChain LLM
llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="vllm-token",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=0.3
)

rewrite_prompt = """
# Role
你是一個 RAG (Retrieval-Augmented Generation) 系統的查詢重寫專家 (Query Rewriter)。
你的任務是將使用者的「最新問題」，結合「對話歷史」，重寫成一個適合讓向量資料庫或搜尋引擎理解的「獨立搜尋語句」。

# Context & Data Knowledge
我們的知識庫包含以下領域的資訊，請在重寫時參考這些背景：
1. **Google Cloud 硬體**：包含 Ironwood (第7代 TPU)、Axion 處理器、C4A (裸機)、N4A (虛擬機器)、GKE 支援等。
2. **AI 開發工具**：Google LiteRT、TensorFlow Lite、NPU/GPU 加速、裝置端推論。
3. **氣象與生活**：台中天氣預報、台灣氣候特徵 (季風/地形雨)、日本旅遊與流感疫情 (H3N2/B型)。

# Rules
1. **指代消解 (Coreference Resolution)**：將「它」、「那個」、「第二個」、「那邊」等代名詞，替換為對話歷史中提到的具體實體 (如：N4A, 台中, LiteRT)。
2. **補全上下文 (Contextualization)**：如果問題簡短（例如「效能如何？」），請補上主詞（例如「Google N4A 虛擬機器的效能如何？」）。
3. **保留原意**：不要回答問題，只要「重寫問題」。不要自行捏造不存在的資訊。
4. **關鍵字增強**：如果使用者的詞彙模糊，請嘗試加入上述 Knowledge 中的專有名詞（例如將「Google 新出的那個 CPU」重寫為包含 `Axion` 或 `Ironwood` 的語句），但不要過度發散。
5. **語言一致性**：輸出必須是繁體中文。

# Output Format
請直接輸出重寫後的搜尋語句，不要包含任何解釋或標點符號以外的文字，絕對禁止輸出任何思考過程、前言、解釋。
"""

def get_embeddings(texts: List[str], task_description: str = "檢索技術文件") -> List[List[float]]:
    """使用 API 獲取文本嵌入向量"""
    response = requests.post(
        EMBEDDING_URL,
        json={
            "texts": texts,
            "normalize": True,
            "batch_size": 32
        }
    )
    return response.json()['embeddings']

def create_collection():
    """建立 Qdrant 集合(支援 Dense + Sparse 混合搜索)"""
    # 檢查集合是否已存在
    try:
        collections = client.get_collections().collections
        if any(c.name == COLLECTION_NAME for c in collections):
            print(f"✓ 集合 '{COLLECTION_NAME}' 已存在,將刪除並重建")
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"檢查集合時發生錯誤: {e}")
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                distance=models.Distance.COSINE,
                size=4096,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )
    print(f"✓ 成功建立集合 '{COLLECTION_NAME}'")


def load_documents(documents: List[str], sources: List[str] = None):
    """
    載入文檔並建立索引
    
    Args:
        documents: 文檔內容列表
        sources: 文檔來源列表 (例如: ["data01.txt", "data02.txt"])
    
    """
    print(f"✓ 載入 {len(documents)} 篇文檔")
    
    # 生成嵌入向量
    print("正在生成嵌入向量...")
    doc_embeddings = get_embeddings(documents, task_description="索引技術文件")
    
    # 準備資料點
    points = [
        models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                "dense": embedding,
                "sparse": models.Document(
                    text=doc,
                    model="Qdrant/bm25",
                ),
            },
            payload={
                "text": doc,
                "source": source  # 加入來源資訊
            },
        )
        for doc, embedding, source in zip(documents, doc_embeddings, sources)
    ]
    
    # 插入向量資料庫
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"✓ 成功將 {len(points)} 篇文檔插入向量資料庫")



def load_documents_from_txt(txt_dir: str):
    """從 txt 檔案載入文檔"""
    import glob
    
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"在 {txt_dir} 中找不到 .txt 檔案")
    
    documents = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                documents.append(content)
    
    load_documents(documents)


def rewrite_query(current_question: str, chat_history: List[Tuple[str, str]]) -> str:
    """使用 LLM 重寫查詢"""
    # 構建對話歷史文本
    history_text = ""
    for q, a in chat_history:
        history_text += f"使用者: {q}\n助理: {a}\n"
    
    # 構建完整 prompt
    full_prompt = f"""{rewrite_prompt}

    # 對話歷史
    {history_text if history_text else "無對話歷史"}

    # 使用者的最新問題
    {current_question}

    # 重寫後的搜尋語句
    """
    
    # 使用 LangChain LLM 呼叫
    response = llm.invoke(full_prompt)
    rewritten_query = response.content.strip()
    
    return rewritten_query



def hybrid_search(query: str, limit: int = 3) -> List[str]:
    """混合搜索(Dense + Sparse)"""
    # 獲取查詢嵌入
    query_embedding = get_embeddings([query], task_description="檢索技術文件")[0]
    
    # 執行混合搜索
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            # BM25 關鍵字搜索
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="sparse",
                limit=limit * 2,
            ),
            # 語義搜索
            models.Prefetch(
                query=query_embedding,
                using="dense",
                limit=limit * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    
    # 提取文檔
    results = [
        (point.payload["text"], point.payload.get("source", "unknown"))
        for point in response.points
    ]
    return results


def generate_answer(question: str, retrieved_results: List[str]) -> str:
        """使用 LLM 根據檢索到的文檔生成答案"""
        # 構建 context
        context_docs = [doc for doc, _ in retrieved_results]
        context = "\n\n---\n\n".join(context_docs)
        
        # 構建 prompt
        prompt = f"""請根據以下檢索到的相關文檔回答使用者的問題。

        # 相關文檔
        {context}

        # 使用者問題
        {question}

        # 回答規則
        - 用繁體中文回答
        - 基於提供的文檔內容
        - 如果文檔中沒有相關資訊,請誠實說明
        - 回答要簡潔明確,不要過度發散
        """
        
        response = llm.invoke(prompt)
        return response.content.strip()


def process_conversation(conversation_id: int, questions: List[str]) -> List[dict]:
    """處理一段對話中的所有問題"""
    results = []
    chat_history = []
    
    for q_id, question in enumerate(questions, start=1):
        print(f"\n{'='*60}")
        print(f"對話 {conversation_id}, 問題 {q_id}: {question}")
        
        # 1. Query Rewrite
        if chat_history:
            rewritten_query = rewrite_query(question, chat_history)
            print(f"重寫後的查詢: {rewritten_query}")
        else:
            rewritten_query = question
            print(f"首個問題,無需重寫: {rewritten_query}")
        
        # 2. Retrieval
        retrieved_docs = hybrid_search(rewritten_query, limit=3)
        print(f"檢索到 {len(retrieved_docs)} 篇相關文檔")
        for i, (doc, source) in enumerate(retrieved_docs, 1):
            print(f"  [{i}] {source}: {doc[:60].replace(chr(10), ' ')}...")
        
        
        # 3. Generate Answer
        answer = generate_answer(question, retrieved_docs)
        print(f"\n答案: {answer[:200]}...")

        source_file = retrieved_docs[0][1] if retrieved_docs else "unknown"
        
        # 4. 記錄結果
        result = {
            'conversation_id': conversation_id,
            'questions_id': q_id,
            'questions': question,
            'answer': rewritten_query,
            'source': source_file
        }
        results.append(result)
        
        # 5. 更新對話歷史
        chat_history.append((question, answer))
    
    return results

#============Main============
"""主程式"""
print("="*60)
print("課堂作業-03: Query Rewrite RAG System")
print("="*60)

# 初始化系統
print("\n初始化系統...")

# 步驟 1: 建立集合
print("\n步驟 1: 建立 Qdrant 集合")
create_collection()

# 步驟 2: 載入文檔
print("\n步驟 2: 載入文檔並建立索引")
upload_dir = "./day6/data"
data_files = [f"data_{i:02d}.txt" for i in range(1, 6)]  # data01.txt ~ data05.txt

documents = []
source = []

for filename in data_files:
    filepath = os.path.join(upload_dir, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
                    source.append(filename)
                    print(f"✓ 載入 {filename}")
        except Exception as e:
            print(f"⚠ 讀取 {filename} 失敗: {e}")
    else:
        print(f"⚠ 找不到 {filename}")

if documents:
    print(f"\n成功載入 {len(documents)} 個檔案: {', '.join(source)}")
    load_documents(documents, source)
else:
    print(f"\n⚠ 未找到 data01.txt ~ data05.txt 檔案")

# 步驟 3: 處理 CSV 中的問題
print("\n步驟 3: 處理 Re_Write_questions.csv")
csv_path = "./day6/Re_Write_questions.csv"
df = pd.read_csv(csv_path)

all_results = []

# 按對話分組處理
for conv_id in sorted(df['conversation_id'].unique()):
    conv_df = df[df['conversation_id'] == conv_id].sort_values('questions_id')
    questions = conv_df['questions'].tolist()
    
    results = process_conversation(conv_id, questions)
    all_results.extend(results)

# 步驟 4: 儲存結果
print("\n步驟 4: 儲存結果到 CSV")
output_df = pd.DataFrame(all_results)

output_path = "./day6/Re_Write_questions_completed.csv"
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✓ 結果已儲存至: {output_path}")