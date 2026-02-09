from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter
import pandas as pd
import os
import re

# ==================== 設定 ====================

client = QdrantClient(url="http://localhost:6333")

EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SCORE_API_URL = "https://hw-01.wade0426.me/submit_answer"

# 🔥 極大切塊 - 確保單一區塊包含完整資訊
CHUNKING_PARAMS = {
    "fixed": {
        "chunk_size": 300,      
        "chunk_overlap": 10,    
        "separator": "。"
    },
    "sliding": {
        "chunk_size": 350,      
        "chunk_overlap": 15,    
        "separators": ["。", "\n\n", "\n"]
    },
    "semantic": {
        "min_size": 150,        
        "max_size": 1300         
    }
}

SEARCH_TOP_K = 10
BATCH_SIZE = 32
API_TIMEOUT = 60

# ==================== 核心函數 ====================

def get_embedding(texts: list) -> tuple:
    """獲取文本嵌入向量"""
    data = {
        "texts": texts,
        "normalize": True,
        "batch_size": BATCH_SIZE
    }
    response = requests.post(EMBED_API_URL, json=data, timeout=API_TIMEOUT)
    return response.json()['embeddings'], response.json()['dimension']

def build_collection(client: QdrantClient, name: str, data: list[dict], dim: int):
    """建立並填充集合"""
    collections = [c.name for c in client.get_collections().collections]
    if name in collections:
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    
    points = []
    total = len(data)
    print(f"   處理 {total} 個區塊...")
    
    for i in range(0, total, BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        texts = [item["text"] for item in batch]
        
        try:
            embeddings, _ = get_embedding(texts)
            
            for j, (item, emb) in enumerate(zip(batch, embeddings)):
                if len(emb) == dim:
                    points.append(PointStruct(
                        id=i+j+1,
                        payload={"text": item["text"], "source": item["source"]},
                        vector=emb
                    ))
            
            print(f"   進度: {min(i+BATCH_SIZE, total)}/{total}", end='\r')
        except Exception as e:
            print(f"\n   ⚠️ 批次 {i} 失敗: {e}")
            continue
    
    if points:
        client.upsert(collection_name=name, points=points)
        print(f"\n   ✅ 建立集合 '{name}': {len(points)} 個向量")

def expand_query(question: str) -> list[str]:
    """
    查詢擴展 - 生成多個查詢變體
    """
    queries = [question]
    
    # 移除疑問詞
    for word in ['何謂', '什麼', '為何', '如何', '哪些', '?', '?']:
        if word in question:
            q = question.replace(word, '').strip()
            if len(q) > 5:
                queries.append(q)
    
    # 提取關鍵詞
    keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', question)
    if keywords:
        sorted_kw = sorted(keywords, key=len, reverse=True)[:3]
        queries.append(' '.join(sorted_kw))
    
    return list(set(queries))[:4]

def search_multi_query(collection_name: str, queries: list[str], top_k: int = 20) -> list:
    """多查詢搜尋並合併結果"""
    all_results = {}
    
    for query in queries:
        try:
            query_vector, _ = get_embedding([query])
            
            search_result = client.query_points(
                collection_name=collection_name,
                query=query_vector[0],
                limit=top_k
            )
            
            for point in search_result.points:
                point_id = point.id
                if point_id not in all_results or point.score > all_results[point_id].score:
                    all_results[point_id] = point
        except:
            continue
    
    sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
    return sorted_results[:top_k]

def select_best_candidate(candidates: list, question: str) -> dict:
    """
    智能選擇最佳候選
    綜合考慮: 相似度、長度、關鍵詞匹配
    """
    if not candidates:
        return None
    
    # 提取問題關鍵詞
    question_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', question))
    
    best_candidate = None
    best_score = -1
    
    for candidate in candidates:
        text = candidate.payload['text']
        similarity = candidate.score
        
        # 綜合評分
        score = 0
        
        # 1. 相似度 (50%)
        score += similarity * 0.5
        
        # 2. 長度獎勵 (30%)
        # 800-1500 字元最佳
        length = len(text)
        if 800 <= length <= 1500:
            length_bonus = 0.3
        elif length > 1500:
            length_bonus = 0.3 * (1 - (length - 1500) / 1000)
        else:
            length_bonus = 0.3 * (length / 800)
        score += max(0, length_bonus)
        
        # 3. 關鍵詞匹配度 (20%)
        text_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', text))
        if question_keywords:
            keyword_overlap = len(question_keywords & text_keywords)
            keyword_ratio = keyword_overlap / len(question_keywords)
            score += keyword_ratio * 0.2
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    return best_candidate

def get_score(q_id: int, retrieve_text: str) -> float:
    """使用 API 獲取評分"""
    try:
        payload = {
            "q_id": int(q_id),
            "student_answer": str(retrieve_text).strip()
        }
        
        response = requests.post(SCORE_API_URL, json=payload, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            return response.json().get('score', 0.0)
        else:
            print(f"   ⚠️ API 錯誤 {response.status_code}")
            return 0.0
    except Exception as e:
        print(f"   ⚠️ 評分異常: {e}")
        return 0.0

# ==================== 切塊方法 ====================

def fixed_size_chunking(text: str, source: str) -> list[dict]:
    """固定大小切塊 - 極大參數"""
    params = CHUNKING_PARAMS["fixed"]
    
    text_splitter = CharacterTextSplitter(
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        separator=params["separator"],
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return [{"text": chunk.strip(), "source": source} 
            for chunk in chunks if len(chunk.strip()) >= 100]

def sliding_window(text: str, source: str) -> list[dict]:
    """滑動視窗切塊 - 極大參數"""
    params = CHUNKING_PARAMS["sliding"]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        separators=params["separators"]
    )
    
    chunks = text_splitter.split_text(text)
    return [{"text": chunk.strip(), "source": source} 
            for chunk in chunks if len(chunk.strip()) >= 100]

def semantic_chunking(text: str, source: str) -> list[dict]:
    """語意切塊 - 極大參數"""
    params = CHUNKING_PARAMS["semantic"]
    
    splitter = TextSplitter((params["min_size"], params["max_size"]))
    chunks = splitter.chunks(text)
    return [{"text": chunk.strip(), "source": source} 
            for chunk in chunks if len(chunk.strip()) >= 100]

# ==================== 主處理流程 ====================

def load_data_files(data_dir: str = "./") -> list[tuple]:
    """載入所有資料檔案"""
    print("\n📂 載入資料檔案...")
    
    data_files = []
    for i in range(1, 6):
        filename = f"data_0{i}.txt"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            data_files.append((content, f"data_0{i}"))
            print(f"   ✅ {filename}: {len(content)} 字元")
        else:
            print(f"   ⚠️ 找不到: {filename}")
    
    return data_files

def process_all_methods(data_files: list[tuple]) -> dict:
    """處理所有切塊方法"""
    print("\n🔄 開始文本切塊 (極大參數)...")
    
    _, dim = get_embedding(['測試'])
    print(f"   ✅ 向量維度: {dim}")
    
    results = {}
    
    methods = [
        ("fixed", fixed_size_chunking, CHUNKING_PARAMS["fixed"]),
        ("sliding", sliding_window, CHUNKING_PARAMS["sliding"]),
        ("semantic", semantic_chunking, CHUNKING_PARAMS["semantic"])
    ]
    
    for method_name, chunking_func, params in methods:
        print(f"\n{'='*60}")
        print(f"📊 處理方法: {method_name}")
        print(f"   參數: {params}")
        print(f"{'='*60}")
        
        all_chunks = []
        for content, source in data_files:
            chunks = chunking_func(content, source)
            all_chunks.extend(chunks)
            print(f"   {source}: {len(chunks)} 個區塊")
        
        print(f"   總計: {len(all_chunks)} 個區塊")
        
        collection_name = f"collection_{method_name}"
        build_collection(client, collection_name, all_chunks, dim)
        results[method_name] = collection_name
    
    return results

def evaluate_questions(collections: dict, questions_df: pd.DataFrame) -> list[dict]:
    """
    評估所有問題 - 使用智能選擇單一最佳結果
    """
    print("\n📝 開始評估問題 (查詢擴展 + 智能選擇)...")
    
    results = []
    record_id = 1
    total_questions = len(questions_df)
    
    for _, row in questions_df.iterrows():
        q_id = row['q_id']
        question = row['questions']
        
        print(f"\n問題 {q_id}/{total_questions}: {question[:50]}...")
        
        # 查詢擴展
        expanded_queries = expand_query(question)
        print(f"   擴展查詢: {len(expanded_queries)} 個變體")
        
        for method_name, collection_name in collections.items():
            # 多查詢搜尋
            candidates = search_multi_query(
                collection_name, 
                expanded_queries,
                top_k=SEARCH_TOP_K
            )
            
            if not candidates:
                print(f"   ⚠️ {method_name}: 無結果")
                continue
            
            # 🔥 智能選擇最佳候選 (不合併)
            best_candidate = select_best_candidate(candidates, question)
            
            if not best_candidate:
                print(f"   ⚠️ {method_name}: 選擇失敗")
                continue
            
            retrieve_text = best_candidate.payload['text']
            source = best_candidate.payload['source']
            similarity = best_candidate.score
            
            # 評分
            score = get_score(q_id, retrieve_text)
            
            results.append({
                'id': record_id,
                'q_id': q_id,
                'method': method_name,
                'retrieve_text': retrieve_text,
                'score': score,
                'source': source
            })
            
            print(f"   {method_name}: 分數={score:.4f}, 相似度={similarity:.4f}, 長度={len(retrieve_text)}, 來源={source}")
            record_id += 1
    
    return results

def save_results_to_csv(results: list[dict], output_file: str = "1411232095_RAG_HW_01.csv"):
    """儲存結果"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 結果已儲存至: {output_file}")
    
    print("\n📊 各方法平均分數:")
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        avg_score = method_df['score'].mean()
        max_score = method_df['score'].max()
        min_score = method_df['score'].min()
        print(f"  {method}: 平均={avg_score:.4f}, 最高={max_score:.4f}, 最低={min_score:.4f}")
    
    return df

def generate_report(df: pd.DataFrame):
    """生成報告"""
    print("\n" + "="*70)
    print("📈 RAG 系統評估報告 (單一結果優化版)")
    print("="*70)
    
    print("\n【1】 參數設定 (極大切塊)")
    print("-" * 70)
    for method_name, params in CHUNKING_PARAMS.items():
        print(f"\n{method_name}:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
    
    print(f"\n【2】 優化策略")
    print("-" * 70)
    print(f"  - 查詢擴展: 生成多個查詢變體")
    
    print("\n【3】 評估結果")
    print("-" * 70)
    stats = df.groupby('method')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(stats.to_string())
    
    print("\n【4】 最佳方法")
    print("-" * 70)
    best_method = df.groupby('method')['score'].mean().idxmax()
    best_score = df.groupby('method')['score'].mean().max()
    print(f"方法: {best_method}")
    print(f"平均分數: {best_score:.4f}")
    
    print("\n" + "="*70)
    return best_score

# ==================== 主程式 ====================

def main():
    """主程式"""
    print("="*70)
    print("🚀 RAG 系統 - 單一結果優化版")
    print("   策略: 極大切塊 + 智能選擇 (不合併)")
    print("="*70)
    
    try:
        data_files = load_data_files()
        if not data_files:
            print("\n❌ 無法載入資料!")
            return
        
        print("\n📋 載入問題...")
        questions_df = pd.read_csv("questions.csv", encoding='utf-8-sig')
        print(f"   ✅ 已載入 {len(questions_df)} 個問題")
        
        collections = process_all_methods(data_files)
        results = evaluate_questions(collections, questions_df)
        df = save_results_to_csv(results)
        generate_report(df)
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()