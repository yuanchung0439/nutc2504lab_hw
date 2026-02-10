import os
import pandas as pd
import requests
import torch
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import uuid
from typing import List, Tuple


# ============ 配置 ============
EMBEDDING_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "day6_hw04_rerank"
RERANKER_MODEL_PATH = os.path.expanduser("./Models/Qwen3-Reranker-0.6B")

client = QdrantClient(url="http://localhost:6333")

# LangChain LLM
llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="vllm-token",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=0.3
)

answer_prompt = """你是一位專業的知識庫助手。請嚴格根據以下提供的【參考資訊】來回答用戶的問題。

### 規則：
1. **必須**只依賴【參考資訊】中的內容回答。
2. 如果【參考資訊】中沒有足夠的資訊來回答問題，請直接回答：「抱歉，根據目前的資料庫，我無法回答這個問題。」，**絕對不要**憑空捏造或使用你的外部知識。
3. 回答應簡潔、準確且專業。
4. 只需回答重點，不用完整句子。

### 參考資訊：
\"\"\"
{context_str}
\"\"\"

### 用戶問題：
{query_str}

現在請開始輸出你的回答"""


# ============ ReRanker 初始化 ============
class ReRanker:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.enabled = False
        self._load_model()
    
    def _load_model(self):
        """載入 ReRanker 模型"""
        try:
            print(f"載入 ReRanker 模型: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                padding_side='left'
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            ).eval()
            
            # Token IDs
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            # 設定
            self.max_length = 8192
            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            
            print("✓ ReRanker 模型載入成功")
            self.enabled = True
            
        except Exception as e:
            print(f"⚠ ReRanker 模型載入失敗: {e}")
            print("將不使用 ReRank 功能")
            self.enabled = False
    
    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """格式化 reranker 輸入"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def process_inputs(self, pairs: List[str]):
        """處理輸入"""
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        inputs = self.tokenizer.pad(
            inputs, 
            padding=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        
        return inputs
    
    @torch.no_grad()
    def compute_scores(self, inputs):
        """計算相關度分數"""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def rerank(
        self, 
        query: str, 
        retrieved_results: List[Tuple[str, str]], 
        task_instruction: str = None
    ) -> List[Tuple[str, str, float]]:
        """
        重新排序文檔
        
        Args:
            query: 查詢
            retrieved_results: [(文檔, 來源), ...]
            task_instruction: 任務指令
        
        Returns:
            [(文檔, 來源, 分數), ...] 按分數降序排序
        """
        if not self.enabled:
            return [(doc, src, 0.0) for doc, src in retrieved_results]
        
        if task_instruction is None:
            task_instruction = '根據查詢檢索相關的技術文件'
        
        documents = [doc for doc, _ in retrieved_results]
        sources = [src for _, src in retrieved_results]
        
        # 格式化輸入
        pairs = [self.format_instruction(task_instruction, query, doc) for doc in documents]
        
        # 計算分數
        inputs = self.process_inputs(pairs)
        scores = self.compute_scores(inputs)
        
        # 組合並排序
        results_with_scores = list(zip(documents, sources, scores))
        results_with_scores.sort(key=lambda x: x[2], reverse=True)
        
        return results_with_scores


# 初始化 ReRanker
reranker = ReRanker(RERANKER_MODEL_PATH)


# ============ 核心功能函數 ============
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
    """載入文檔並建立索引"""
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
                "source": source
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


def hybrid_search(query: str, limit: int = 10) -> List[Tuple[str, str]]:
    """
    混合搜索(Dense + Sparse)
    
    Args:
        limit: 檢索數量 (用於 ReRank 時建議 10-20)
    """
    query_embedding = get_embeddings([query], task_description="檢索技術文件")[0]
    
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="sparse",
                limit=limit,
            ),
            models.Prefetch(
                query=query_embedding,
                using="dense",
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    
    results = [
        (point.payload["text"], point.payload.get("source", "unknown"))
        for point in response.points
    ]
    return results


def generate_answer(question: str, reranked_results: List[Tuple[str, str, float]]) -> str:
    """使用 LLM 根據檢索到的文檔生成答案"""
    # 只取文檔內容
    context_docs = [doc for doc, _, _ in reranked_results]
    context = "\n\n---\n\n".join(context_docs)
    
    # 使用新的 prompt
    prompt = answer_prompt.format(context_str=context, query_str=question)
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠ LLM 生成答案失敗: {e}")
        return f"根據檢索到的文檔,{context_docs[0][:200]}..."


def process_questions(questions_df: pd.DataFrame) -> List[dict]:
    """處理所有問題"""
    results = []
    
    for idx, row in questions_df.iterrows():
        q_id = row['題目_ID']
        question = row['題目']
        
        print(f"\n{'='*60}")
        print(f"問題 {q_id}: {question}")
        
        # 1. Hybrid Search (檢索 Top-10 候選)
        retrieved_results = hybrid_search(question, limit=10)
        print(f"✓ Hybrid Search 檢索到 {len(retrieved_results)} 篇候選文檔")
        
        # 2. ReRank (重新排序)
        reranked_results = reranker.rerank(question, retrieved_results)
        print(f"✓ ReRank 完成,取 Top-3:")
        
        # 只取 Top-3
        top3_results = reranked_results[:3]
        for i, (doc, source, score) in enumerate(top3_results, 1):
            print(f"  [{i}] {source} (分數: {score:.4f}): {doc[:50].replace(chr(10), ' ')}...")
        
        # 3. Generate Answer
        answer = generate_answer(question, top3_results)
        print(f"答案: {answer[:150]}...")
        
        # 4. 取第一個(最相關的)來源
        source_file = top3_results[0][1] if top3_results else "unknown"
        
        # 5. 記錄結果
        result = {
            '題目_ID': q_id,
            '題目': question,
            '標準答案': answer,
            '來源文件': source_file
        }
        results.append(result)
    
    return results


# ============ Main ============
def main():
    print("="*60)
    print("課堂作業-04: RAG System with ReRank")
    print("="*60)
    
    # 步驟 1: 建立集合
    print("\n步驟 1: 建立 Qdrant 集合")
    create_collection()
    
    # 步驟 2: 載入文檔
    print("\n步驟 2: 載入文檔並建立索引")
    upload_dir = "./day6/data"
    data_files = [f"data_{i:02d}.txt" for i in range(1, 6)]
    
    documents = []
    sources = []
    
    for filename in data_files:
        filepath = os.path.join(upload_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        documents.append(content)
                        sources.append(filename)
                        print(f"✓ 載入 {filename}")
            except Exception as e:
                print(f"⚠ 讀取 {filename} 失敗: {e}")
        else:
            print(f"⚠ 找不到 {filename}")
    
    if documents:
        print(f"\n成功載入 {len(documents)} 個檔案: {', '.join(sources)}")
        load_documents(documents, sources)
    else:
        print(f"\n⚠ 未找到 data_01.txt ~ data_05.txt 檔案")
        return
    
    # 步驟 3: 處理 questions.csv
    print("\n步驟 3: 處理 questions.csv")
    csv_path = "./cw04/questions.csv"
    df = pd.read_csv(csv_path)
    
    # 處理所有問題
    all_results = process_questions(df)
    
    # 步驟 4: 儲存結果
    print("\n步驟 4: 儲存結果到 CSV")
    output_df = pd.DataFrame(all_results)
    
    output_path = "./cw04/questions_completed_hw04.csv"
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 結果已儲存至: {output_path}")
    
    # 顯示結果預覽
    print("\n" + "="*60)
    print("結果預覽:")
    print("="*60)
    for idx, row in output_df.iterrows():
        print(f"\n[{row['題目_ID']}] {row['題目']}")
        print(f"來源: {row['來源文件']}")
        print(f"答案: {row['標準答案'][:100]}...")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()