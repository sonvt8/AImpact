#!/bin/bash
set -eu

jwt_secret_file=/app/data/.jwt_secret

if [ -z "${JWT_SECRET:-}" ]; then
  mkdir -p /app/data
  if [ -s "$jwt_secret_file" ]; then
    JWT_SECRET="$(tr -d '\r\n' < "$jwt_secret_file")"
  else
    JWT_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
    umask 077
    printf '%s\n' "$JWT_SECRET" > "$jwt_secret_file"
  fi
  chmod 600 "$jwt_secret_file"
  export JWT_SECRET
fi

if [ ! -f "${MODEL_PATH:-/app/models/cc.vi.300.bin}" ]; then
  printf '%s\n' \
    'LỖI: Không tìm thấy FastText model.' \
    'Cần đặt FastText model tại ./models/cc.vi.300.bin (~4.5GB) trước khi chạy.' >&2
  exit 78
fi

if [ "${ADMIN_PASSWORD:-}" = 'ChangeMe-Dev-2026' ]; then
  printf '%s\n' \
    '============================================================' \
    'CẢNH BÁO DEV: ĐỔI mật khẩu & đặt JWT_SECRET riêng cho production.' \
    '============================================================' >&2
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
