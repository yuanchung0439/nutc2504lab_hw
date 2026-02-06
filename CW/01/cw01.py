from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
import requests

COLLECTION_NAME = "test_collection_2"

client = QdrantClient(url="http://localhost:6333")



response = requests.post(
    "https://ws-04.wade0426.me/embed",
    json={
        "texts": ["人工智慧是什麼？", "深度學習的應用"],
        "task_description": "檢索技術文件",
        "normalize": True
    }
)

def create_collection(name: str, distance: Distance, dimension=response.json()["dimension"]):
    client.create_collection(
        collection_name= name,
        vectors_config=VectorParams(size=dimension, distance=distance),
    )

def get_embedding(texts: list):
    data = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }
    response = requests.post("https://ws-04.wade0426.me/embed", json=data)

    return response.json()['embeddings']



def upsert_vector(text: list) :
    print(f"狀態碼: {response.status_code}")
    print(f"回應内容: {response.text}")

    if response.status_code == 200:
        result = response.json()
        print(f"Dimension: {result['dimension']}")
    else:
        print(f"Error: {response.json()}")


    for t in enumerate(text):
        client.upsert(
            collection_name= COLLECTION_NAME,
            points=[
                PointStruct(
                    id=t[0]+1,
                    vector=get_embedding(t[1])[0],
                    payload={"text": t[1], "metadata": "其他資源"}
                )
            ]
        )

upsert_vector([["人工智慧很有趣"], ["AI"], ["Apple"], ["深度學習"], ["人工智慧的應用"]])
texts = ["AI 有麽好處?"]
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