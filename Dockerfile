# Utilisez une image de base NVIDIA CUDA avec Python
# C'est CRUCIAL. Choisissez la version CUDA qui correspond à votre version de PyTorch (ex: 12.1 pour cu121).
# Les images `runtime` sont plus petites et contiennent le nécessaire pour faire tourner des applications CUDA.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Installez Python et pip dans cette image de base
# Vérifiez la version de python3 disponible dans l'image de base (ex: python3.10 ou python3.11)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Définir Python 3 comme le binaire par défaut
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # Spécifiez --extra-index-url pour PyTorch GPU si votre version n'est pas sur PyPI par défaut
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copier le code de l'application
COPY app.py .

# Exposer le port par défaut de Cloud Run
ENV PORT=8080

# Commande pour lancer l'application avec Gunicorn (s'assure que gunicorn est dans requirements.txt)
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "app:app"]