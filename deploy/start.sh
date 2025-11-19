#!/bin/bash

# NeMo App - Smart Startup Script with Hot Reload Preservation
# Usage: ./deploy/start.sh --gpu    (for GPU mode)
#        ./deploy/start.sh --cpu    (for CPU mode)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_PORTS=(5173 8000 8002 5432)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

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

# Function to display usage
usage() {
    echo "Usage: $0 [--gpu|--cpu] [--qdrant|--pgvector]"
    echo ""
    echo "Options:"
    echo "  --gpu      Start with GPU-accelerated vLLM (requires NVIDIA GPU)"
    echo "  --cpu      Start with CPU-only vLLM (slower, but works without GPU)"
    echo "  --qdrant   Use Qdrant as vector store backend"
    echo "  --pgvector Use PostgreSQL/pgvector as vector store backend (default)"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu"
    echo "  $0 --cpu --qdrant"
    echo "  $0 --gpu --pgvector"
    exit 1
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if a port is in use
port_in_use() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    elif netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Get process using a port
get_port_process() {
    local port=$1
    lsof -Pi :$port -sTCP:LISTEN -t 2>/dev/null || echo ""
}

# Check if a specific container is running and healthy
is_container_healthy() {
    local container_name=$1

    # Check if container exists and is running
    if ! docker ps --filter "name=$container_name" --format '{{.Names}}' | grep -q "^${container_name}$"; then
        return 1  # Container not running
    fi

    # For containers with healthcheck, verify health status
    local health_status=$(docker inspect "$container_name" --format='{{.State.Health.Status}}' 2>/dev/null)

    if [ -z "$health_status" ]; then
        # No healthcheck defined, just check if running
        return 0
    fi

    # Check health status
    if [ "$health_status" = "healthy" ]; then
        return 0
    else
        return 1
    fi
}

# Get container health status for display
get_container_status() {
    local container_name=$1

    if ! docker ps -a --filter "name=$container_name" --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo "not_found"
        return
    fi

    local state=$(docker inspect "$container_name" --format='{{.State.Status}}' 2>/dev/null)

    if [ "$state" != "running" ]; then
        echo "$state"
        return
    fi

    local health_status=$(docker inspect "$container_name" --format='{{.State.Health.Status}}' 2>/dev/null)

    if [ -z "$health_status" ]; then
        echo "running"
    else
        echo "$health_status"
    fi
}

# Check if all required containers are healthy
check_containers_health() {
    cd "$PROJECT_ROOT"

    # Get the actual project name from docker compose
    local project_name=$(docker compose -f "$COMPOSE_FILE" config --format json 2>/dev/null | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)

    if [ -z "$project_name" ]; then
        # Fallback to directory-based naming
        project_name=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    fi

    local all_healthy=true
    local any_running=false
    local services=("postgres" "backend" "frontend")

    # Add vLLM service based on mode
    if [ "$MODE" = "gpu" ]; then
        services+=("vllm-gpu")
    else
        services+=("vllm-cpu")
    fi

    print_info "Checking container health..."
    echo ""

    for service in "${services[@]}"; do
        local container_name="${project_name}-${service}-1"
        local status=$(get_container_status "$container_name")

        case "$status" in
            "healthy")
                print_success "$service is healthy"
                any_running=true
                ;;
            "running")
                print_success "$service is running (no healthcheck)"
                any_running=true
                ;;
            "not_found")
                print_warning "$service not found"
                all_healthy=false
                ;;
            "starting"|"unhealthy")
                print_warning "$service is $status"
                any_running=true
                all_healthy=false
                ;;
            *)
                print_error "$service is $status"
                all_healthy=false
                ;;
        esac
    done

    echo ""

    if [ "$all_healthy" = true ] && [ "$any_running" = true ]; then
        return 0  # All containers healthy
    elif [ "$any_running" = true ]; then
        return 1  # Some containers running but not all healthy
    else
        return 2  # No containers running
    fi
}

# Prerequisites check
check_prerequisites() {
    print_header "Checking Prerequisites"

    local checks_passed=true

    # Check Docker
    print_info "Checking Docker installation..."
    if command_exists docker; then
        print_success "Docker is installed ($(docker --version))"
    else
        print_error "Docker is not installed. Please install Docker first."
        checks_passed=false
    fi

    # Check Docker Compose
    print_info "Checking Docker Compose installation..."
    if docker compose version >/dev/null 2>&1; then
        print_success "Docker Compose is installed ($(docker compose version))"
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        checks_passed=false
    fi

    # Check if Docker daemon is running
    print_info "Checking Docker daemon..."
    if docker info >/dev/null 2>&1; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running. Please start Docker."
        checks_passed=false
    fi

    if [ "$checks_passed" = false ]; then
        print_error "Prerequisites check failed. Please fix the issues above."
        exit 1
    fi

    print_success "All prerequisites satisfied!"
}

# Check for port conflicts
check_ports() {
    print_header "Checking Port Availability"

    local external_conflicts=false
    local nemo_containers_using_ports=false

    for port in "${REQUIRED_PORTS[@]}"; do
        if port_in_use $port; then
            local pid=$(get_port_process $port)

            # Check if it's one of our containers
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "nemo-app\|nemo_app"; then
                print_info "Port $port is in use by NeMo App container (this is fine)"
                nemo_containers_using_ports=true
            else
                print_warning "Port $port is in use by external process (PID: $pid)"
                external_conflicts=true
            fi
        else
            print_success "Port $port is available"
        fi
    done

    # Warn about external conflicts
    if [ "$external_conflicts" = true ]; then
        print_warning "External port conflicts detected"
        print_info "This may prevent services from starting correctly"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Aborted by user"
            exit 1
        fi
    fi
}

# Build and start services
start_services() {
    local rebuild=$1  # "full", "partial", or "none"

    cd "$PROJECT_ROOT"

    if [ "$rebuild" = "full" ]; then
        print_header "Building and Starting All Services"

        print_info "Building Docker images in parallel..."
        docker compose -f "$COMPOSE_FILE" build --parallel
        print_success "Images built successfully"
        echo ""

        print_info "Starting services in parallel (detached mode)..."
        print_info "Active profile: $COMPOSE_PROFILES"
        docker compose -f "$COMPOSE_FILE" up -d
        print_success "Services started"

    elif [ "$rebuild" = "partial" ]; then
        print_header "Restarting Unhealthy Services"

        print_info "Rebuilding unhealthy services..."
        docker compose -f "$COMPOSE_FILE" build --parallel
        print_success "Images built"
        echo ""

        print_info "Restarting services (preserving healthy containers)..."
        docker compose -f "$COMPOSE_FILE" up -d --no-recreate
        print_success "Services updated"

    else
        print_header "All Services Already Healthy"
        print_success "No restart needed - hot reload is active!"
        print_info "Your code changes will be automatically detected"
    fi
}

# Wait for service health
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=1

    print_info "Waiting for $service_name to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            print_success "$service_name is healthy"
            return 0
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo
    print_error "$service_name failed to become healthy after $max_attempts attempts"
    return 1
}

# Monitor service health after startup
monitor_services() {
    print_header "Monitoring Service Health"

    # Wait for Backend (checks both PostgreSQL and vLLM)
    if ! wait_for_service "Backend API" "http://localhost:8000/health/live" 30; then
        print_warning "Backend may still be starting. Check logs with: docker compose logs backend"
    fi

    # Wait for vLLM (slower to start, especially on first run)
    if ! wait_for_service "vLLM API" "http://localhost:8002/v1/models" 60; then
        print_warning "vLLM may still be loading the model. This can take several minutes on first startup."
    fi

    # Wait for Backend full health (after vLLM is ready)
    if ! wait_for_service "Backend (full)" "http://localhost:8000/health" 20; then
        print_warning "Backend health check incomplete. Check logs for details."
    fi

    # Check Frontend (simple port check)
    print_info "Waiting for Frontend to be ready..."
    local attempt=1
    while [ $attempt -le 30 ]; do
        if port_in_use 5173; then
            print_success "Frontend is ready"
            break
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo
}

# Show service status
show_status() {
    print_header "Service Status"

    cd "$PROJECT_ROOT"
    docker compose -f "$COMPOSE_FILE" ps

    print_header "Service URLs"
    echo -e "${GREEN}Frontend:${NC}    http://localhost:5173"
    echo -e "${GREEN}Backend:${NC}     http://localhost:8000"
    echo -e "${GREEN}API Docs:${NC}    http://localhost:8000/docs"
    echo -e "${GREEN}PostgreSQL:${NC}  localhost:5432 (nemo_rag database)"
    echo -e "${GREEN}vLLM API:${NC}    http://localhost:8002/v1/models"
    if [ "$VECTOR_STORE_BACKEND" = "qdrant" ]; then
        echo -e "${GREEN}Qdrant:${NC}      http://localhost:6333/dashboard"
    fi

    print_header "Vector Store Configuration"
    if [ "$VECTOR_STORE_BACKEND" = "qdrant" ]; then
        echo -e "${GREEN}Backend:${NC}     Qdrant"
        echo -e "${GREEN}Dashboard:${NC}   http://localhost:6333/dashboard"
    else
        echo -e "${GREEN}Backend:${NC}     PostgreSQL/pgvector"
    fi

    print_header "Hot Reload Status"
    echo -e "${GREEN}✓${NC} Frontend: Hot reload enabled (Vite HMR)"
    echo -e "${GREEN}✓${NC} Backend: Hot reload enabled (uvicorn --reload)"
    echo -e "${BLUE}ℹ${NC} Edit files in ./frontend or ./backend and see changes automatically"

    print_header "Useful Commands"
    echo -e "${BLUE}View logs:${NC}         docker compose logs -f"
    echo -e "${BLUE}View backend logs:${NC} docker compose logs -f backend"
    echo -e "${BLUE}View frontend logs:${NC} docker compose logs -f frontend"
    echo -e "${BLUE}Stop services:${NC}     ./deploy/stop.sh"
    echo -e "${BLUE}Force restart:${NC}     docker compose down && ./deploy/start.sh $MODE_FLAG"
}

# Main execution
main() {
    print_header "NeMo App - Smart Startup with Hot Reload"

    # Parse arguments
    if [ $# -eq 0 ]; then
        echo "Error: No mode specified."
        usage
    fi

    MODE=""
    MODE_FLAG=""
    VECTOR_BACKEND="pgvector"  # Default to pgvector

    # Parse all arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu)
                MODE="gpu"
                MODE_FLAG="--gpu"
                ;;
            --cpu)
                MODE="cpu"
                MODE_FLAG="--cpu"
                ;;
            --qdrant)
                VECTOR_BACKEND="qdrant"
                ;;
            --pgvector)
                VECTOR_BACKEND="pgvector"
                ;;
            *)
                echo "Error: Invalid option '$1'"
                usage
                ;;
        esac
        shift
    done

    # Check if mode was specified
    if [ -z "$MODE" ]; then
        echo "Error: No compute mode specified (--gpu or --cpu required)."
        usage
    fi

    # Display configuration
    if [ "$MODE" = "gpu" ]; then
        print_info "Starting with GPU-accelerated vLLM"
    else
        print_info "Starting with CPU-only vLLM"
        print_warning "Note: CPU mode will be slower than GPU mode"
    fi

    if [ "$VECTOR_BACKEND" = "qdrant" ]; then
        print_info "Using Qdrant as vector store backend"
    else
        print_info "Using PostgreSQL/pgvector as vector store backend"
    fi

    # Set the Docker Compose profiles
    export COMPOSE_PROFILES="fullstack,$MODE"

    # Set vector store backend environment variable
    export VECTOR_STORE_BACKEND="$VECTOR_BACKEND"

    # Run checks
    check_prerequisites
    check_ports

    # Check current container health
    local health_check_result
    check_containers_health || health_check_result=$?
    health_check_result=${health_check_result:-0}

    local rebuild_mode="none"

    if [ $health_check_result -eq 0 ]; then
        # All containers healthy - no rebuild needed
        print_success "All containers are already running and healthy!"
        print_info "Hot reload is active. No restart needed."
        rebuild_mode="none"
    elif [ $health_check_result -eq 1 ]; then
        # Some containers running but not all healthy
        print_warning "Some containers need attention"
        rebuild_mode="partial"
    else
        # No containers running - full startup needed
        print_info "No containers running. Starting from scratch..."
        rebuild_mode="full"
    fi

    # Start or restart services as needed
    start_services "$rebuild_mode"

    # Monitor health only if we started/restarted services
    if [ "$rebuild_mode" != "none" ]; then
        monitor_services
    fi

    # Show final status
    show_status

    # Offer to show logs
    if [ "$rebuild_mode" != "none" ]; then
        echo ""
        read -p "Follow container logs? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            print_info "Following logs... (Press Ctrl+C to stop)"
            echo ""
            docker compose -f "$COMPOSE_FILE" logs -f
        fi
    fi

    print_success "Ready for development!"
}

# Run main function
main "$@"
