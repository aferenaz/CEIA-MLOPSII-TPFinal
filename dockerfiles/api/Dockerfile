FROM python:3.11-slim
WORKDIR /app
COPY dockerfiles/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/gout_common ./gout_common
COPY api/app ./app
EXPOSE 8800
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8800"]
