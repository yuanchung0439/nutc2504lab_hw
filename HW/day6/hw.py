import os
import requests
import pandas as pd
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import uuid
from qdrant_client import QdrantClient, models
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

# ============================================================================
# 配置參數
# ============================================================================

class Config:
    """配置類別"""
    # Qdrant
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "day6_water_company_kb"
    
    # LLM API
    LLM_BASE_URL = "https://ws-06.huannago.com/v1"
    LLM_MODEL = "gemma-3-27b-it"
    LLM_TEMPERATURE = 0.3
    
    # Embedding API
    EMBEDDING_URL = "https://ws-04.wade0426.me/embed"
    EMBEDDING_DIM = 4096
    
    # Reranker 模型路徑 (本地)
    RERANKER_MODEL_PATH = os.path.expanduser("./Models/Qwen3-Reranker-0.6B")
    
    # 檔案路徑
    QA_DATA_PATH = "./day6-c/qa_data.txt"
    QUESTIONS_CSV = "./day6-c/day6_HW_questions.csv"
    OUTPUT_CSV = "./day6-c/outputs/day6_HW_questions_completed.csv"
    
    # 檢索參數
    INITIAL_SEARCH_LIMIT = 5  # 初始檢索數量
    RERANK_TOP_K = 3  # Rerank 後保留的文件數量
    
    # DeepEval 參數
    DEEPEVAL_THRESHOLD = 0.5

# ============================================================================
# Embedding API
# ============================================================================

def get_embeddings(texts: List[str]) -> List[List[float]]:

    response = requests.post(
        Config.EMBEDDING_URL,
        json={
            "texts": texts,
        "normalize": True,
        "batch_size": 32
        }
    )
    
    return response.json()['embeddings']

# ============================================================================
# Qdrant 向量資料庫
# ============================================================================

class QdrantManager:
    """Qdrant 管理類別"""
    
    def __init__(self):
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self.collection_name = Config.COLLECTION_NAME
    
    def create_collection(self):
        """建立支援 Hybrid Search 的集合"""
        try:
            # 檢查集合是否存在
            collections = self.client.get_collections().collections
            if any(col.name == self.collection_name for col in collections):
                print(f"✅ 集合 '{self.collection_name}' 已存在，刪除舊的...")
                self.client.delete_collection(self.collection_name)
            
            # 建立新集合
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        distance=models.Distance.COSINE,
                        size=Config.EMBEDDING_DIM,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
            print(f"✅ 成功建立集合 '{self.collection_name}'")
        except Exception as e:
            print(f"❌ 建立集合失敗: {e}")
            raise
    
    def load_documents(self, qa_data_path: str):
        """載入知識庫"""
        print(f"\n載入知識庫: {qa_data_path}")
        
        # 讀取文件
        with open(qa_data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按問答對分割
        chunks = []
        current_chunk = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('來源：'):
                current_chunk.append(line)
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) > 20:  # 過濾太短的
                        chunks.append(chunk_text)
                current_chunk = []
            elif line:
                current_chunk.append(line)
        
        # 加入最後一個 chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) > 20:
                chunks.append(chunk_text)
        
        print(f"✓ 分割為 {len(chunks)} 個文檔片段")
        
        # 生成嵌入
        print("正在生成嵌入向量...")
        embeddings = get_embeddings(chunks)
        
        # 建立索引
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "dense": embedding,
                    "sparse": models.Document(
                        text=chunk,
                        model="Qdrant/bm25",
                    ),
                },
                payload={
                    "text": chunk,
                    "chunk_id": idx + 1
                }
            ))
        
        # 插入向量資料庫
        self.client.upsert(
            collection_name=Config.COLLECTION_NAME,
            points=points
        )
        print(f"✓ 成功插入 {len(points)} 個文檔片段到向量資料庫")
    
    def hybrid_search(self, query: str, limit: int = Config.INITIAL_SEARCH_LIMIT) -> List[str]:
        """
        Hybrid Search (Dense + Sparse/BM25)
        
        Args:
            query: 查詢字串
            limit: 檢索數量
        
        Returns:
            文件列表
        """
        try:
            # 取得 query embedding
            query_embedding = get_embeddings([query])[0]
            
            # Hybrid Search with RRF
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # BM25 關鍵字搜索
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model="Qdrant/bm25",
                        ),
                        using="sparse",
                        limit=limit,
                    ),
                    # 語義搜索
                    models.Prefetch(
                        query=query_embedding,
                        using="dense",
                        limit=limit,
                    ),
                ],
                # 使用 RRF 融合演算法
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )
            
            # 提取文件
            documents = [point.payload["text"] for point in response.points]
            return documents
            
        except Exception as e:
            print(f"❌ Hybrid Search 失敗: {e}")
            raise

# ============================================================================
# Reranker
# ============================================================================

class Reranker:
    """Reranker 類別 (Qwen3-Reranker-0.6B)"""
    
    def __init__(self):
        print("🔄 載入 Reranker 模型...")
        
        # 載入模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.RERANKER_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            padding_side='left'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.RERANKER_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        ).eval()
        
        # 配置參數
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        # Prompt 模板
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        print("✅ Reranker 模型載入完成")
    
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
    
    def rerank(self, query: str, documents: List[str], 
               task_instruction: str = None, top_k: int = Config.RERANK_TOP_K) -> List[Tuple[str, float]]:
        """
        重新排序文件
        
        Args:
            query: 查詢字串
            documents: 文件列表
            task_instruction: 任務指令
            top_k: 返回前 k 個結果
        
        Returns:
            (文件, 分數) 元組列表
        """
        if not documents:
            return []
        
        if task_instruction is None:
            task_instruction = '根據使用者問題，找出最相關的台水公司客服資訊'
        
        # 格式化輸入
        pairs = [self.format_instruction(task_instruction, query, doc) for doc in documents]
        
        # 計算分數
        inputs = self.process_inputs(pairs)
        scores = self.compute_scores(inputs)
        
        # 組合並排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]

# ============================================================================
# LLM (用於生成答案和 DeepEval)
# ============================================================================

class CustomLLM(DeepEvalBaseLLM):
    """自訂 LLM 類別 (用於 DeepEval)"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key="NoNeed",
            base_url="https://ws-02.wade0426.me/v1"
        )
        self.model_name = "local-model"
    
    def load_model(self):
        return self.client
    
    def generate(self, prompt: str) -> str:
        """生成回應"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=Config.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self):
        return self.model_name

class RAGSystem:
    """RAG 系統"""
    
    def __init__(self, qdrant_manager: QdrantManager, reranker: Reranker):
        self.qdrant = qdrant_manager
        self.reranker = reranker
        self.llm = CustomLLM()
    
    def query_rewrite(self, query: str) -> str:
        """Query Rewrite - 將使用者問題改寫為更適合檢索的形式"""
        prompt = f"""請將以下使用者問題改寫為更適合檢索的查詢語句。
        保持問題的核心意圖，但使用更精確的關鍵詞。

        原始問題：{query}

        改寫後的查詢（只輸出改寫結果，不要其他說明）："""
        
        rewritten_query = self.llm.generate(prompt)
        return rewritten_query.strip()
    
    def retrieve_documents(self, query: str, use_rewrite: bool = True) -> Tuple[List[str], str]:
        """
        檢索相關文件
        
        Args:
            query: 原始查詢
            use_rewrite: 是否使用 query rewrite
        
        Returns:
            (檢索到的文件列表, 實際使用的查詢)
        """
        # Query Rewrite
        if use_rewrite:
            search_query = self.query_rewrite(query)
            print(f"  🔄 Query Rewrite: {query} → {search_query}")
        else:
            search_query = query
        
        # Hybrid Search
        print(f"  🔍 Hybrid Search...")
        candidate_docs = self.qdrant.hybrid_search(search_query, Config.INITIAL_SEARCH_LIMIT)
        print(f"  📄 找到 {len(candidate_docs)} 個候選文件")
        
        # Rerank
        print(f"  🎯 Reranking...")
        reranked_results = self.reranker.rerank(search_query, candidate_docs, top_k=Config.RERANK_TOP_K)
        
        documents = [doc for doc, score in reranked_results]
        print(f"  ✅ 最終保留 {len(documents)} 個文件")
        
        return documents, search_query
    
    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """根據檢索到的文件生成答案"""
        context = "\n\n".join([f"[文件 {i+1}]\n{doc}" for i, doc in enumerate(context_docs)])
        
        prompt = f"""你是台灣自來水公司的AI客服助手。請根據以下提供的參考文件回答使用者的問題。

        參考文件：
        {context}

        使用者問題：{query}

        1. 請根據參考文件提供準確、完整且易懂的回答。如果參考文件中沒有足夠資訊，請誠實告知。
        2. 答案請簡潔，寫成一行完整回答

        回答："""
        
        answer = self.llm.generate(prompt)
        return answer.strip()
    
    def answer_query(self, query: str) -> Dict[str, any]:
        """
        完整的問答流程
        
        Returns:
            包含 query, answer, retrieval_contexts 的字典
        """
        print(f"\n{'='*80}")
        print(f"❓ 使用者問題: {query}")
        
        # 檢索文件
        documents, search_query = self.retrieve_documents(query)
        
        # 生成答案
        print(f"  💬 生成答案...")
        answer = self.generate_answer(query, documents)
        
        print(f"  ✅ 答案: {answer[:100]}...")
        
        return {
            "query": query,
            "answer": answer,
            "retrieval_contexts": documents,
            "search_query": search_query
        }

# ============================================================================
# DeepEval 評估
# ============================================================================

def evaluate_with_deepeval(result: Dict, expected_answer: str, custom_llm: CustomLLM) -> Dict[str, float]:
    """
    使用 DeepEval 評估 RAG 系統
    
    Args:
        result: RAG 系統的輸出結果
        expected_answer: 預期答案 (ground truth)
        custom_llm: 自訂 LLM
    
    Returns:
        各項指標的分數
    """
    print(f"\n  📊 DeepEval 評估中...")
    
    # 建立測試案例
    test_case = LLMTestCase(
        input=result["query"],
        actual_output=result["answer"],
        expected_output=expected_answer,
        retrieval_context=result["retrieval_contexts"]
    )
    
    # 定義指標
    metrics = {
        "Faithfulness": FaithfulnessMetric(
            threshold=Config.DEEPEVAL_THRESHOLD,
            model=custom_llm,
            include_reason=False
        ),
        "Answer_Relevancy": AnswerRelevancyMetric(
            threshold=Config.DEEPEVAL_THRESHOLD,
            model=custom_llm,
            include_reason=False
        ),
        "Contextual_Recall": ContextualRecallMetric(
            threshold=Config.DEEPEVAL_THRESHOLD,
            model=custom_llm,
            include_reason=False
        ),
        "Contextual_Precision": ContextualPrecisionMetric(
            threshold=Config.DEEPEVAL_THRESHOLD,
            model=custom_llm,
            include_reason=False
        ),
        "Contextual_Relevancy": ContextualRelevancyMetric(
            threshold=Config.DEEPEVAL_THRESHOLD,
            model=custom_llm,
            include_reason=False
        )
    }
    
    # 評估各項指標
    scores = {}
    for metric_name, metric in metrics.items():
        try:
            metric.measure(test_case)
            scores[metric_name] = metric.score
            print(f"    {metric_name}: {metric.score:.4f}")
        except Exception as e:
            print(f"    ⚠️ {metric_name} 評估失敗: {e}")
            scores[metric_name] = None
    
    return scores

# ============================================================================
# 主程式
# ============================================================================

def main():
    """主程式"""
    print("=" * 80)
    print("台水公司 AI 客服助手 - Day 6 作業")
    print("=" * 80)
    
    # 建立輸出目錄
    os.makedirs(os.path.dirname(Config.OUTPUT_CSV), exist_ok=True)
    
    # 1. 初始化 Qdrant
    print("\n📦 初始化 Qdrant...")
    qdrant_manager = QdrantManager()
    qdrant_manager.create_collection()
    qdrant_manager.load_documents(Config.QA_DATA_PATH)
    
    # 2. 初始化 Reranker
    print("\n🎯 初始化 Reranker...")
    reranker = Reranker()
    
    # 3. 初始化 RAG 系統
    print("\n🤖 初始化 RAG 系統...")
    rag_system = RAGSystem(qdrant_manager, reranker)
    
    # 4. 讀取問題資料
    print(f"\n📖 讀取問題資料: {Config.QUESTIONS_CSV}")
    df_questions = pd.read_excel(Config.QUESTIONS_CSV)
    
    # **只處理前5筆**
    df_questions = df_questions.head(5)
    print(f"📝 處理前 {len(df_questions)} 筆問題")
    
    # 讀取參考答案
    qa_answer_path = Config.QUESTIONS_CSV.replace("day6_HW_questions.csv", "questions_answer.csv")
    df_answers = pd.read_excel(qa_answer_path)
    
    # 5. 處理每個問題
    results = []
    
    for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="處理問題"):
        q_id = row['q_id']
        question = row['questions']
        
        # 取得參考答案
        expected_answer = df_answers[df_answers['q_id'] == q_id]['answer'].values[0]
        
        print(f"\n{'='*80}")
        print(f"處理問題 {q_id}/{len(df_questions)}")
        
        try:
            # RAG 問答
            result = rag_system.answer_query(question)
            
            # DeepEval 評估
            scores = evaluate_with_deepeval(result, expected_answer, rag_system.llm)
            
            # 記錄結果
            results.append({
                'q_id': q_id,
                'questions': question,
                'answer': result['answer'],
                'Faithfulness': scores.get('Faithfulness'),
                'Answer_Relevancy': scores.get('Answer_Relevancy'),
                'Contextual_Recall': scores.get('Contextual_Recall'),
                'Contextual_Precision': scores.get('Contextual_Precision'),
                'Contextual_Relevancy': scores.get('Contextual_Relevancy')
            })
            
        except Exception as e:
            print(f"❌ 處理問題 {q_id} 時發生錯誤: {e}")
            results.append({
                'q_id': q_id,
                'questions': question,
                'answer': "錯誤：無法生成答案",
                'Faithfulness': None,
                'Answer_Relevancy': None,
                'Contextual_Recall': None,
                'Contextual_Precision': None,
                'Contextual_Relevancy': None
            })
    
    # 6. 儲存結果
    print(f"\n💾 儲存結果到: {Config.OUTPUT_CSV}")
    df_results = pd.DataFrame(results)
    df_results.to_csv(Config.OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    # 7. 顯示統計
    print("\n" + "=" * 80)
    print("📊 評估結果統計")
    print("=" * 80)
    
    for metric in ['Faithfulness', 'Answer_Relevancy', 'Contextual_Recall', 
                   'Contextual_Precision', 'Contextual_Relevancy']:
        scores = df_results[metric].dropna()
        if len(scores) > 0:
            print(f"{metric:25s}: 平均 {scores.mean():.4f} | 最小 {scores.min():.4f} | 最大 {scores.max():.4f}")
        else:
            print(f"{metric:25s}: 無有效數據")
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()