# 1. 파이썬 3.12 설치
FROM python:3.12-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (가끔 필요한 경우가 있음)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . .

# 6. 서버 실행 (FastAPI)
# 주의: app.main:app 부분은 실제 main.py 위치에 맞춰야 함
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]