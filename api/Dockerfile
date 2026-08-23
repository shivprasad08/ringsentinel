# RingSentinel API — containerized serving layer.
# Data (the data/ directory produced by the pipeline script) is
# deliberately NOT baked into this image — it's a separate, regenerable
# artifact (retrain weekly, redeploy never, or vice versa). Mount it as
# a volume at runtime instead. This is standard practice for splitting
# a model-serving container from its training/batch pipeline, and it's
# also just more honest about what's "the app" vs. "the data."

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# data/ is expected to be mounted at runtime:
#   docker run -v $(pwd)/data:/app/data -p 8000:8000 ringsentinel-api
ENV RINGSENTINEL_DATA_DIR=/app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
