# Image de base CUDA avec Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Installation Python + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Lien symbolique python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Répertoire de travail
WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copier le code
COPY app.py .

# Définir le port
ENV PORT=8080

CMD gunicorn --bind :$PORT --workers 1 --threads 8 app:app
