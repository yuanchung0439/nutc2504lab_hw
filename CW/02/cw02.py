from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
import requests
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter
import pandas as pd
import re, tiktoken

client = QdrantClient(url="http://localhost:6333")
response = requests.post(
    "https://ws-04.wade0426.me/embed",
    json={
        "texts": ["人工智慧是什麼？", "深度學習的應用"],
        "task_description": "檢索技術文件",
        "normalize": True
    }
)

COLLECTION_NAME = "t02_collection"

with open('./day5/text.txt', "r", encoding='utf-8') as f:
    text = f.read()

def markdown_to_csv(md_file,csv_file):
    with open(md_file, "r", encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    data = []

    for line in lines:
        if re.match(r'^\|?[\\s\-:|]+\|?$', line):
            continue
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')]
            cells = [c for c in cells if c]
            data.append(cells)

    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_csv(csv_file, index=False, encoding='utf-8')

#markdown_to_csv('./day5/table/table_txt.md', './day5/table/output.csv')

#tables = pd.read_html("./day5/table/table_html.html", encoding='utf-8')
#print(tables[0])

def insert_VDB(id:int, text: list, tokens: int):
    client.upsert(
        collection_name= COLLECTION_NAME,
        points=[
            PointStruct(
                id=id,
                payload={"text": text, "num_tokens": tokens},
                vector=get_embedding(text)[0]
            )
        ]
    )

def get_embedding(texts: list):
    data = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }
    response = requests.post("https://ws-04.wade0426.me/embed", json=data)

    return response.json()['embeddings']

def get_points(texts: list):
    query_vector = get_embedding(texts)[0]

    search_result = client.query_points(
        collection_name= COLLECTION_NAME,
        query=query_vector,
        limit=5
    )

    for point in search_result.points:
        print(f"ID: {point.id}")
        print(f"相似度分數(Score): {point.score}")
        print(f"内容: {point.payload['text']}")
        print("---")


def character_splitter():
    text_splitter = CharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separator="。",
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    print(f"總共産生 {len(chunks)} 個分塊\n")

    for i, chunk in enumerate(chunks, 1):
        insert_VDB(i, [chunk.strip()], len(chunk))
        print(f"=== 分塊 {i} ===")
        print(f"長度:{len(chunk)} 字符")
        print(f"内容: {chunk.strip()}")
        print()
    get_points(chunks)

character_splitter()

def token_splitter():
    text_splitter = TokenTextSplitter(
        chunk_size =100,
        chunk_overlap=10,
        model_name="gpt-4"
    )

    chunks = text_splitter.split_text(text)
    print(f"原始文本長度: {len(text)} tokens")
    print(f"分塊數量: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        insert_VDB(i, [chunk.strip()], len(chunk))
        print(f"分塊 {i+1}:")
        print(f" 長度: {len(chunk)} tokens\n")
        print(f"内容: {chunk[:50]}")
        print()
    get_points(chunks)

def recursive_splitter(): 
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=80,
        chunk_overlap=10,
        separators=[""]
    )
    encoding = tiktoken.encoding_for_model("gpt-4")
    chunks = text_splitter.split_text(text)

    print(f"原始文本長度: {len(text)} tokens")
    print(f"分塊數量: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        insert_VDB(i, [chunk.strip()], len(chunk))
        token_count = len(encoding.encode(chunk))
        print(f"分塊 {i+1}:")
        print(f" 長度: {token_count} tokens\n")
        print(f"内容: {chunk[:50]}")
        print()