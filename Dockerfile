FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY src/internal_assets/weights/v11_l.pt /app/src/weights/v11_l.pt
WORKDIR /app/src
RUN mkdir -p weights utils
ENV PORT=9100
ENV HOST=0.0.0.0
ENV WEIGHT_PATH=./weights
ENV DATA_YAML_PATH=./utils/args.yaml
ENV CONF_THR=0.25
ENV IOU_THR=0.05
EXPOSE 9100
CMD ["uvicorn", "jang:app", "--host", "0.0.0.0", "--port", "9100"]
