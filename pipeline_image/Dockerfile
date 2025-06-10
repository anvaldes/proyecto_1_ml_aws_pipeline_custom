# Imagen base mínima y compatible con SageMaker
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /opt/ml/code

# Copiar requirements y scripts
COPY requirements.txt ./
COPY evaluate.py ./evaluate.py
COPY preprocessing.py ./preprocessing.py

# Instalar las dependencias del proyecto
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 🔥 Declarar el punto de entrada para SageMaker
ENTRYPOINT ["python3"]