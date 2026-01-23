FROM python:3.11-slim as builder
WORKDIR /build
RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# PDF 생성을 위해 필요한 시스템 라이브러리 추가 (선택 사항: libpango 등 필요한 경우)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 1. 소스 코드 및 디렉토리 복사
COPY *.py ./
COPY results_transformer_4feat/ ./results_transformer_4feat/
COPY rag_corpus/ ./rag_corpus/

# 2. 폰트 파일 복사 (가장 중요한 부분!)
# 로컬의 NanumGothic-Regular.ttf 파일을 컨테이너의 /app 경로로 복사합니다.
COPY NanumGothic-Regular.ttf ./

# 3. 권한 설정
# 폰트 파일을 포함하여 모든 파일의 권한을 appuser에게 부여합니다.
RUN mkdir -p rag_index && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 9999

# HEALTHCHECK (선택 사항: 원본의 CMD 구문을 헬스체크로 사용 가능)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#   CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9999/')" || exit 1

CMD ["python", "RAG_server.py"]