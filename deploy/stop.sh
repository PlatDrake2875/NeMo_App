#!/bin/bash

# NeMo App - Graceful Shutdown Script
# This script stops all running services cleanly

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
PROFILE_FLAGS=(--profile fullstack --profile gpu --profile cpu)

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Stop services
stop_services() {
    print_header "Stopping NeMo App Services"

    cd "$PROJECT_ROOT"

    # Stop all possible profiles (fullstack, gpu, cpu)
    # This ensures we stop everything regardless of how it was started
    local all_profiles="fullstack,gpu,cpu"

    # Check if any services are running
    if docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" ps -q 2>/dev/null | grep -q .; then
        print_info "Stopping services gracefully..."

        # Stop services (gives containers time to shut down)
        docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" stop
        print_success "Services stopped"

        # Remove containers
        print_info "Removing containers..."
        docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" down
        print_success "Containers removed"
    else
        print_warning "No running services found"
    fi
}

# Show cleanup options
show_cleanup_options() {
    print_header "Cleanup Options"

    echo -e "${BLUE}What would you like to clean up?${NC}"
    echo -e "1) Keep data volumes (default)"
    echo -e "2) Remove data volumes (PostgreSQL database and vLLM cache will be lost)"
    echo -e "3) Remove volumes and images"
    echo

    read -p "Choose an option (1-3) [1]: " -n 1 -r
    echo

    OPTION=${REPLY:-1}

    case $OPTION in
        1)
            print_info "Keeping data volumes"
            ;;
        2)
            print_warning "Removing data volumes..."
            cd "$PROJECT_ROOT"
            docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" down -v
            print_success "Data volumes removed (PostgreSQL, vLLM cache, Qdrant)"
            ;;
        3)
            print_warning "Removing volumes and images..."
            cd "$PROJECT_ROOT"
            docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" down -v --rmi local
            print_success "Volumes and images removed"
            ;;
        *)
            print_info "Invalid option, keeping data volumes"
            ;;
    esac
}

# Show final status
show_status() {
    print_header "Final Status"

    cd "$PROJECT_ROOT"

    # Check if any containers are still running
    if docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" ps -q 2>/dev/null | grep -q .; then
        print_warning "Some containers are still running:"
        docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" ps
    else
        print_success "All NeMo App services have been stopped"
    fi

    # Check for remaining volumes
    local volumes=$(docker volume ls --filter name=nemo -q 2>/dev/null | wc -l)
    if [ "$volumes" -gt 0 ]; then
        print_info "Data volumes are preserved (PostgreSQL, vLLM cache, Qdrant)"
        print_info "To remove volumes, run: docker compose down -v"
    fi

    print_header "Restart Services"
    echo -e "${BLUE}To start services again:${NC} ./deploy/start.sh --gpu"
    echo -e "${BLUE}With Qdrant:${NC} ./deploy/start.sh --gpu --qdrant"
}

docker_cleanup() {
    cd "$PROJECT_ROOT"
    docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" down

    # Only remove THIS project's networks (not all networks!)
    local project_name
    project_name=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    docker network ls --filter "name=${project_name}" -q 2>/dev/null | xargs -r docker network rm 2>/dev/null || true
}


# Main execution
main() {
    print_header "NeMo App - Graceful Shutdown"

    stop_services
    show_cleanup_options
    show_status

    print_success "Shutdown complete!"
}

# Run main function
main
