# 1. 파이썬 베이스 이미지 (3.10도 좋지만, 로컬 환경인 3.13-slim 권장)
FROM python:3.13-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치
# 한글 폰트 렌더링 지원을 위해 libffi 등이 필요할 수 있습니다.
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 모든 파일 복사 (NanumGothic-Regular.ttf 포함)
COPY . .

# 6. 포트 설정 (로컬에서 9999 쓰셨으니 9999로 맞추는 게 편합니다)
EXPOSE 9999

# 7. 실행 명령어
# 로컬에서 실행하던 방식 그대로 uvicorn을 내부에서 호출하는 RAG_server.py 실행
CMD ["python", "RAG_server.py"]