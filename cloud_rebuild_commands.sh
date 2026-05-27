# Cloud mode: full rebuild without cache
#
# 1. Stop and remove all containers (cloud mode)
docker compose --env-file .env.cloud --env-file .env.secrets -f docker-compose.yml -f docker-compose.cloud.yml down --remove-orphans

# 2. Build all images from scratch (no cache, pull latest base images)
docker compose --env-file .env.cloud --env-file .env.secrets -f docker-compose.yml -f docker-compose.cloud.yml build --no-cache --pull

# 3. Start all services with fresh containers (force recreate)
docker compose --env-file .env.cloud --env-file .env.secrets -f docker-compose.yml -f docker-compose.cloud.yml up -d --force-recreate --remove-orphans
