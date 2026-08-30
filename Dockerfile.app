# Dockerfile.app — builds the Streamlit frontend service
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

# API_URL should be set as an env var at deploy time to point at the
# deployed FastAPI service's public URL (not localhost, once both are
# running as separate deployed services).
CMD ["sh", "-c", "streamlit run src/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
