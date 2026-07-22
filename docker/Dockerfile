FROM node:22-alpine AS web-build
WORKDIR /build/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AIMPACT_ROOT=/app
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api/ ./api/
COPY project_agent/ ./project_agent/
COPY --from=web-build /build/web/dist ./web/dist/
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod 0755 ./docker/entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/app/docker/entrypoint.sh"]
