FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face provides PORT (usually 7860). Use it.
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
# Or: CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]