#!/bin/bash

# NeMo App - Universal Deployment Wrapper
# Usage: ./deploy/universal.sh <command> [options]
# Commands: start, stop, status, logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
PROFILE_FLAGS=(--profile fullstack --profile gpu --profile cpu)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    echo "NeMo App - Universal Deployment Wrapper"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [--gpu|--cpu] [--qdrant|--pgvector]  Start services"
    echo "  stop                                        Stop all services"
    echo "  status                                      Show service status"
    echo "  logs [service]                              Follow logs (all or specific service)"
    echo ""
    echo "Examples:"
    echo "  $0 start --gpu              # Start with GPU acceleration"
    echo "  $0 start --cpu --qdrant     # Start CPU mode with Qdrant"
    echo "  $0 status                   # Show running services"
    echo "  $0 logs backend             # Follow backend logs"
    echo "  $0 stop                     # Stop all services"
    exit 1
}

cmd_start() {
    # Validate that --gpu or --cpu is provided
    local has_mode=false
    for arg in "$@"; do
        case "$arg" in
            --gpu|--cpu)
                has_mode=true
                ;;
        esac
    done

    if [ "$has_mode" = false ]; then
        echo -e "${RED}Error: --gpu or --cpu flag is required${NC}"
        echo ""
        echo "Usage: $0 start --gpu    # For GPU-accelerated mode"
        echo "       $0 start --cpu    # For CPU-only mode"
        exit 1
    fi

    exec "$SCRIPT_DIR/start.sh" "$@"
}

cmd_stop() {
    exec "$SCRIPT_DIR/stop.sh"
}

cmd_status() {
    echo -e "${BLUE}NeMo App Service Status${NC}"
    echo ""
    cd "$PROJECT_ROOT"
    docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" ps
}

cmd_logs() {
    cd "$PROJECT_ROOT"
    if [ $# -eq 0 ]; then
        # Follow all logs
        docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" logs -f
    else
        # Follow specific service logs
        docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" logs -f "$@"
    fi
}

# Main entry point
if [ $# -eq 0 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    start)
        cmd_start "$@"
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        echo ""
        usage
        ;;
esac
