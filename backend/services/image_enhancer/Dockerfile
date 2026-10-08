FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /SlimDiffSR_API

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
    && tar -xzf Python-3.10.12.tgz \
    && cd Python-3.10.12 \
    && ./configure --enable-optimizations \
    && make -j"$(nproc)" \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.10.12 Python-3.10.12.tgz

RUN python3.10 --version
RUN python3.10 -m pip --version

COPY requirements.txt .

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3.10", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]