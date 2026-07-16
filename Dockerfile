FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace/AudioDec

RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/AudioDec/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 torchaudio==2.7.0 \
    && python -m pip install -r /workspace/AudioDec/requirements.txt

COPY . /workspace/AudioDec
RUN chmod +x /workspace/AudioDec/scripts/docker/*.sh
