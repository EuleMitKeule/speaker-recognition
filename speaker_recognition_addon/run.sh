#!/usr/bin/env sh
# Home Assistant add-on entrypoint for Speaker Recognition.
#
# Reads the add-on options from /data/options.json (using the Python already
# present in the base image, so no bashio/jq dependency is needed) and maps them
# to the environment variables the server expects, then launches the service.
set -e

OPTIONS="/data/options.json"

get_opt() {
    # $1 = option key, $2 = default value
    python -c "import json,sys; print(json.load(open('$OPTIONS')).get('$1', '$2'))" 2>/dev/null || printf '%s' "$2"
}

if [ -f "$OPTIONS" ]; then
    HOST="$(get_opt host '0.0.0.0')"
    PORT="$(get_opt port '8099')"
    LOG_LEVEL="$(get_opt log_level 'info')"
    ACCESS_LOG="$(get_opt access_log 'true')"
    EMBEDDINGS_DIR="$(get_opt embeddings_dir '/share/speaker_recognition/embeddings')"
else
    HOST="0.0.0.0"
    PORT="8099"
    LOG_LEVEL="info"
    ACCESS_LOG="true"
    EMBEDDINGS_DIR="/share/speaker_recognition/embeddings"
fi

# The server expects an upper-case log level.
LOG_LEVEL="$(printf '%s' "$LOG_LEVEL" | tr '[:lower:]' '[:upper:]')"

export HOST PORT LOG_LEVEL ACCESS_LOG EMBEDDINGS_DIR

echo "[speaker-recognition] Starting server: host=${HOST} port=${PORT} log_level=${LOG_LEVEL} embeddings_dir=${EMBEDDINGS_DIR}"

mkdir -p "${EMBEDDINGS_DIR}"

cd /app
exec python -m speaker_recognition
