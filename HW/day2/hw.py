from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import time

llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="vllm-token",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=0,
    max_tokens=30
)
topic = input("Input Topic: ")
parser = StrOutputParser()
chain_a = ChatPromptTemplate.from_template(f"LinkedIn style chinese story in 30 words {topic} ") | llm | parser
chain_b = ChatPromptTemplate.from_template(f"Instagram style chinese story in 30 words {topic} ") | llm | parser

combined_chain = RunnableParallel(linkedin=chain_a, instagram=chain_b)


start_time = time.time()

for chunk in combined_chain.stream({"topic": topic}):
    print(chunk,end="\n", flush=True)

end_time = time.time()

print(f"Timing: {end_time - start_time:.4f} seconds")

results = combined_chain.batch([
    {"topic": topic}
])
for i, res in enumerate(results):
    print(res, end="\n")