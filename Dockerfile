FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt
COPY Script.py .

ENTRYPOINT ["python3", "Script.py"]
