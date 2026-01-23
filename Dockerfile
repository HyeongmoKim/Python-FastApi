FROM python:3.11-slim as builder
WORKDIR /build
RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 모든 파일 복사 (원본 그대로)
COPY *.py ./
COPY results_transformer_4feat/ ./results_transformer_4feat/
COPY rag_corpus/ ./rag_corpus/

RUN mkdir -p rag_index && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# 원본 포트 9999 사용
EXPOSE 9999

    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9999/')" || exit 1

# 원본 그대로 실행
CMD ["python", "RAG_server.py"]
