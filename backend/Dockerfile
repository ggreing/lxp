# 1. Base Image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app
ENV PYTHONPATH /app

# 3. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY . .

# 5. Expose the port the app runs on
EXPOSE 8001

# 6. Command to run the application
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8001"]
