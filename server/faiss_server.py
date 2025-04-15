# faiss_pdf_mcp.py
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from dotenv import load_dotenv


# API KEY 정보로드
load_dotenv()


# PDF 로더 & 임베딩 모델 준비
PDF_PATH = "./data/langgraph_adaptive_rag.pdf"  # ← 여기에 분석할 PDF 파일 경로를 넣으세요
loader = PDFPlumberLoader(file_path=PDF_PATH)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 문서 로드 
docs = loader.load()

# FAISS 인덱스 생성
faiss_index = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)

# 검색기 생성
retriever = faiss_index.as_retriever(search_kwargs={"k": 8})

# MCP 인스턴스 생성
mcp = FastMCP("faiss_retriever", host="0.0.0.0", port=8005)

# MCP 툴 정의
@mcp.tool()
def retrieve(query: str) -> List[str]:

    results = retriever.invoke(query)
    # Document 객체의 .page 메타데이터와 .page_content를 함께 반환
    return [
        f"[page {doc.metadata.get('page')}] {doc.page_content}"
        for doc in results
    ]

# 7) 서버 실행
if __name__ == "__main__":
    # stdio 방식으로 MCP 서버 실행
    mcp.run(transport="stdio")
