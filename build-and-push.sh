#!/bin/bash
# Build and push Docker images to Docker Hub
# Usage: ./build-and-push.sh [version_tag]
# Example: ./build-and-push.sh v1.0.0
# If no version tag provided, uses 'latest'

set -e  # Exit on error

# Configuration
DOCKER_USERNAME="namelesshampus"
AI_SERVICE_IMAGE="${DOCKER_USERNAME}/radiator-ai-service"
BOT_IMAGE="${DOCKER_USERNAME}/radiator-bot"
VERSION_TAG="${1:-latest}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Smart Radiator Assistant - Build and Push ===${NC}"
echo -e "Version tag: ${GREEN}${VERSION_TAG}${NC}"
echo ""

# Check if logged in to Docker Hub
if ! docker info | grep -q "Username: ${DOCKER_USERNAME}"; then
    echo -e "${RED}Not logged in to Docker Hub. Please run: docker login${NC}"
    exit 1
fi

# Build AI Service
echo -e "${BLUE}Building AI Service...${NC}"
docker build -t "${AI_SERVICE_IMAGE}:${VERSION_TAG}" ./ai_service
if [ "$VERSION_TAG" != "latest" ]; then
    docker tag "${AI_SERVICE_IMAGE}:${VERSION_TAG}" "${AI_SERVICE_IMAGE}:latest"
fi
echo -e "${GREEN}✓ AI Service built successfully${NC}"
echo ""

# Build Bot
echo -e "${BLUE}Building Bot...${NC}"
docker build -t "${BOT_IMAGE}:${VERSION_TAG}" ./bot
if [ "$VERSION_TAG" != "latest" ]; then
    docker tag "${BOT_IMAGE}:${VERSION_TAG}" "${BOT_IMAGE}:latest"
fi
echo -e "${GREEN}✓ Bot built successfully${NC}"
echo ""

# Push AI Service
echo -e "${BLUE}Pushing AI Service...${NC}"
docker push "${AI_SERVICE_IMAGE}:${VERSION_TAG}"
if [ "$VERSION_TAG" != "latest" ]; then
    docker push "${AI_SERVICE_IMAGE}:latest"
fi
echo -e "${GREEN}✓ AI Service pushed successfully${NC}"
echo ""

# Push Bot
echo -e "${BLUE}Pushing Bot...${NC}"
docker push "${BOT_IMAGE}:${VERSION_TAG}"
if [ "$VERSION_TAG" != "latest" ]; then
    docker push "${BOT_IMAGE}:latest"
fi
echo -e "${GREEN}✓ Bot pushed successfully${NC}"
echo ""

echo -e "${GREEN}=== All images built and pushed successfully ===${NC}"
echo ""
echo "Images pushed:"
echo "  - ${AI_SERVICE_IMAGE}:${VERSION_TAG}"
echo "  - ${BOT_IMAGE}:${VERSION_TAG}"
if [ "$VERSION_TAG" != "latest" ]; then
    echo "  - ${AI_SERVICE_IMAGE}:latest"
    echo "  - ${BOT_IMAGE}:latest"
fi
echo ""
echo "To deploy, run: docker compose up -d"
