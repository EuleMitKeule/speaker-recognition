ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG TEMPIO_VERSION BUILD_ARCH
RUN \
    curl -sSLf -o /usr/bin/tempio \
    "https://github.com/home-assistant/tempio/releases/download/${TEMPIO_VERSION}/tempio_${BUILD_ARCH}"

WORKDIR /app

RUN apk add --no-cache \
    python3 \
    py3-pip \
    gcc \
    g++ \
    musl-dev \
    python3-dev \
    cmake \
    ninja \
    llvm15-dev \
    llvm15-static

COPY pyproject.toml ./
COPY speaker_recognition ./speaker_recognition

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -e ".[server]"

ENV HOST=0.0.0.0 \
    PORT=8099 \
    LOG_LEVEL=INFO \
    ACCESS_LOG=true \
    EMBEDDINGS_DIR=/share/speaker_recognition/embeddings

COPY rootfs /

RUN chmod a+x /etc/s6-overlay/s6-rc.d/speaker-recognition/run

CMD ["/init"]
