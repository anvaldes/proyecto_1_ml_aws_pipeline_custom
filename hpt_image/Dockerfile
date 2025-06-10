# Imagen base mínima y compatible con SageMaker
FROM python:3.10-slim

# Establecer directorio de trabajo requerido por SageMaker
ENV WORKDIR=/opt/ml/code
WORKDIR $WORKDIR

# Copiar archivos del proyecto al contenedor
COPY train.py .
COPY requirements.txt .

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar las dependencias del proyecto
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 🔥 Declarar el punto de entrada para SageMaker
ENTRYPOINT ["python", "train.py"]

