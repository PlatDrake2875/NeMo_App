"""
Agent router for NeMo Guardrails agent management.
Handles agent metadata and configuration.
"""

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


def load_agent_metadata() -> dict:
    """Load agent metadata from YAML file."""
    try:
        metadata_path = (
            Path(__file__).parent.parent / "guardrails_config" / "metadata.yaml"
        )

        if not metadata_path.exists():
            logger.warning("Metadata file not found, using defaults")
            return {
                "agents": [
                    {
                        "name": "Math Assistant",
                        "directory": "math_assistant",
                        "description": "Specialized in mathematics, equations, and mathematical concepts",
                        "icon": "🧮",
                        "persona": "Martin Scorsese-inspired math specialist",
                    },
                    {
                        "name": "Bank Assistant",
                        "directory": "bank_assistant",
                        "description": "Expert in banking, financial services, and account management",
                        "icon": "🏦",
                        "persona": "Professional banking advisor",
                    },
                    {
                        "name": "Aviation Assistant",
                        "directory": "aviation_assistant",
                        "description": "Specialist in flight operations, aircraft systems, and aviation",
                        "icon": "✈️",
                        "persona": "Aviation operations expert",
                    },
                ]
            }

        with open(metadata_path, encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

        # Validate that required fields exist
        if not metadata or "agents" not in metadata:
            raise ValueError("Invalid metadata structure")

        for agent in metadata["agents"]:
            required_fields = ["name", "directory", "description"]
            for field in required_fields:
                if field not in agent:
                    raise ValueError(
                        f"Missing required field '{field}' in agent metadata"
                    )

        return metadata

    except Exception as e:
        logger.error("Error loading agent metadata: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to load agent metadata: {str(e)}"
        ) from e


@router.get("/metadata")
async def get_agent_metadata():
    """
    Get metadata for all available NeMo Guardrails agents.

    Returns:
        Dict containing agent information including names, directories, descriptions, and icons.
    """
    try:
        metadata = load_agent_metadata()
        logger.info(
            "Successfully loaded metadata for %d agents",
            len(metadata.get("agents", [])),
        )
        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in get_agent_metadata: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/available")
async def get_available_agents():
    """
    Get list of available agent directories.

    Returns:
        List of agent directory names that actually exist on the filesystem.
    """
    try:
        config_dir = Path(__file__).parent.parent / "guardrails_config"

        if not config_dir.exists():
            raise HTTPException(
                status_code=404, detail="Guardrails configuration directory not found"
            )

        # Find all subdirectories that contain both config.yml and config.co
        available_agents = []

        for item in config_dir.iterdir():
            if item.is_dir() and item.name not in ["__pycache__"]:
                config_yml = item / "config.yml"
                config_co = item / "config.co"

                if config_yml.exists() and config_co.exists():
                    available_agents.append(item.name)

        logger.info(
            "Found %d available agents: %s", len(available_agents), available_agents
        )

        return {"agents": available_agents}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting available agents: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to get available agents: {str(e)}"
        ) from e


@router.get("/validate/{agent_name}")
async def validate_agent(agent_name: str):
    """
    Validate that an agent configuration exists and is properly configured.

    Args:
        agent_name: Name of the agent directory to validate

    Returns:
        Dict with validation status and details.
    """
    try:
        config_dir = Path(__file__).parent.parent / "guardrails_config" / agent_name

        validation_result = {
            "agent_name": agent_name,
            "exists": False,
            "config_yml_exists": False,
            "config_co_exists": False,
            "valid": False,
            "errors": [],
        }

        if not config_dir.exists():
            validation_result["errors"].append(
                f"Agent directory '{agent_name}' does not exist"
            )
            return validation_result

        validation_result["exists"] = True

        config_yml = config_dir / "config.yml"
        config_co = config_dir / "config.co"

        validation_result["config_yml_exists"] = config_yml.exists()
        validation_result["config_co_exists"] = config_co.exists()

        if not config_yml.exists():
            validation_result["errors"].append("config.yml file missing")

        if not config_co.exists():
            validation_result["errors"].append("config.co file missing")

        # If both files exist, try to load and validate the YAML
        if config_yml.exists():
            try:
                with open(config_yml, encoding="utf-8") as f:
                    yaml_content = yaml.safe_load(f)

                # Basic validation of YAML structure
                if not isinstance(yaml_content, dict):
                    validation_result["errors"].append(
                        "config.yml is not a valid YAML object"
                    )
                elif "models" not in yaml_content:
                    validation_result["errors"].append(
                        "config.yml missing required 'models' section"
                    )

            except yaml.YAMLError as e:
                validation_result["errors"].append(f"Invalid YAML syntax: {str(e)}")
            except Exception as e:
                validation_result["errors"].append(
                    f"Error reading config.yml: {str(e)}"
                )

        validation_result["valid"] = len(validation_result["errors"]) == 0

        logger.info(
            "Agent validation for '%s': %s",
            agent_name,
            "valid"
            if validation_result["valid"]
            else f"invalid - {validation_result['errors']}",
        )

        return validation_result

    except Exception as e:
        logger.error("Error validating agent '%s': %s", agent_name, e)
        raise HTTPException(
            status_code=500, detail=f"Validation error: {str(e)}"
        ) from e


# Custom agents directory
CUSTOM_AGENTS_DIR = Path(__file__).parent.parent / "guardrails_config" / "custom"


def get_all_agents():
    """Get all agents with metadata, including custom ones."""
    config_dir = Path(__file__).parent.parent / "guardrails_config"
    agents = []

    # Load built-in agent metadata
    try:
        metadata = load_agent_metadata()
        for agent in metadata.get("agents", []):
            agent_dir = config_dir / agent.get("directory", agent.get("name"))
            if agent_dir.exists():
                agents.append({
                    "name": agent.get("directory", agent.get("name")),
                    "display_name": agent.get("name"),
                    "description": agent.get("description", ""),
                    "icon": agent.get("icon", "🤖"),
                    "is_custom": False,
                })
    except Exception as e:
        logger.warning("Error loading agent metadata: %s", e)

    # Load custom agents
    if CUSTOM_AGENTS_DIR.exists():
        for item in CUSTOM_AGENTS_DIR.iterdir():
            if item.is_dir() and (item / "config.yml").exists():
                # Try to read description from config.yml
                description = ""
                try:
                    with open(item / "config.yml", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                        description = config.get("description", "")
                except Exception:
                    pass

                agents.append({
                    "name": item.name,
                    "display_name": item.name.replace("_", " ").title(),
                    "description": description,
                    "icon": "⚙️",
                    "is_custom": True,
                })

    return agents


@router.get("")
async def list_agents():
    """
    List all available agents with their metadata.

    Returns:
        Dict containing list of agents with names, descriptions, and custom status.
    """
    try:
        agents = get_all_agents()
        return {"agents": agents}
    except Exception as e:
        logger.error("Error listing agents: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{agent_name}/config")
async def get_agent_config(agent_name: str):
    """
    Get the configuration files for an agent.

    Args:
        agent_name: Name of the agent directory

    Returns:
        Dict with config_yaml and config_colang content.
    """
    try:
        config_dir = Path(__file__).parent.parent / "guardrails_config"

        # Check custom agents first
        agent_dir = CUSTOM_AGENTS_DIR / agent_name
        is_custom = True

        if not agent_dir.exists():
            agent_dir = config_dir / agent_name
            is_custom = False

        if not agent_dir.exists():
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

        config_yml_path = agent_dir / "config.yml"
        config_co_path = agent_dir / "config.co"

        config_yaml = ""
        config_colang = ""
        metadata = {}

        if config_yml_path.exists():
            with open(config_yml_path, encoding="utf-8") as f:
                config_yaml = f.read()
                try:
                    metadata = yaml.safe_load(config_yaml) or {}
                except Exception:
                    pass

        if config_co_path.exists():
            with open(config_co_path, encoding="utf-8") as f:
                config_colang = f.read()

        return {
            "config_yaml": config_yaml,
            "config_colang": config_colang,
            "metadata": metadata,
            "is_custom": is_custom,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting config for agent '%s': %s", agent_name, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{agent_name}/config")
async def save_agent_config(agent_name: str, config: dict):
    """
    Save the configuration files for a custom agent.

    Args:
        agent_name: Name of the agent directory
        config: Dict with config_yaml and config_colang content

    Returns:
        Success message.
    """
    try:
        # Only allow saving custom agents
        agent_dir = CUSTOM_AGENTS_DIR / agent_name

        if not agent_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Can only edit custom agents. Clone this agent first."
            )

        config_yaml = config.get("config_yaml", "")
        config_colang = config.get("config_colang", "")

        # Validate YAML before saving
        try:
            yaml.safe_load(config_yaml)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")

        # Save files
        with open(agent_dir / "config.yml", "w", encoding="utf-8") as f:
            f.write(config_yaml)

        with open(agent_dir / "config.co", "w", encoding="utf-8") as f:
            f.write(config_colang)

        logger.info("Saved config for agent '%s'", agent_name)
        return {"message": "Configuration saved successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error saving config for agent '%s': %s", agent_name, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("")
async def create_agent(agent_data: dict):
    """
    Create a new custom agent.

    Args:
        agent_data: Dict with name, description, and optional base_agent

    Returns:
        Success message with agent name.
    """
    try:
        name = agent_data.get("name", "").strip()
        description = agent_data.get("description", "")
        base_agent = agent_data.get("base_agent")

        if not name:
            raise HTTPException(status_code=400, detail="Agent name is required")

        # Sanitize name
        safe_name = name.lower().replace(" ", "_")

        # Create custom agents directory if needed
        CUSTOM_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

        agent_dir = CUSTOM_AGENTS_DIR / safe_name

        if agent_dir.exists():
            raise HTTPException(status_code=400, detail="Agent already exists")

        agent_dir.mkdir(parents=True)

        # If cloning from base agent
        if base_agent:
            config_dir = Path(__file__).parent.parent / "guardrails_config"
            base_dir = config_dir / base_agent

            if not base_dir.exists():
                base_dir = CUSTOM_AGENTS_DIR / base_agent

            if base_dir.exists():
                # Copy config files
                import shutil
                for file in ["config.yml", "config.co"]:
                    src = base_dir / file
                    if src.exists():
                        shutil.copy(src, agent_dir / file)
        else:
            # Create default config files
            default_yaml = f"""# Configuration for {name}
# Description: {description}

models:
  - type: main
    engine: ollama
    model: llama3.2:latest

instructions:
  - type: general
    content: |
      You are a helpful AI assistant named {name}.
      {description}
"""
            default_colang = f"""# Colang configuration for {name}

define user greeting
  "hello"
  "hi"
  "hey"

define bot greeting
  "Hello! How can I help you today?"

define flow greeting
  user greeting
  bot greeting
"""

            with open(agent_dir / "config.yml", "w", encoding="utf-8") as f:
                f.write(default_yaml)

            with open(agent_dir / "config.co", "w", encoding="utf-8") as f:
                f.write(default_colang)

        logger.info("Created new agent '%s'", safe_name)
        return {"message": "Agent created successfully", "name": safe_name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating agent: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{agent_name}")
async def delete_agent(agent_name: str):
    """
    Delete a custom agent.

    Args:
        agent_name: Name of the agent to delete

    Returns:
        Success message.
    """
    try:
        agent_dir = CUSTOM_AGENTS_DIR / agent_name

        if not agent_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Agent not found or cannot delete built-in agents"
            )

        # Delete the directory
        import shutil
        shutil.rmtree(agent_dir)

        logger.info("Deleted agent '%s'", agent_name)
        return {"message": "Agent deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting agent '%s': %s", agent_name, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{agent_name}/validate-content")
async def validate_agent_content(agent_name: str, content: dict):
    """
    Validate agent configuration content before saving.

    Args:
        agent_name: Name of the agent
        content: Dict with config_yaml and config_colang content

    Returns:
        Validation result with errors if any.
    """
    try:
        errors = []
        config_yaml = content.get("config_yaml", "")
        config_colang = content.get("config_colang", "")

        # Validate YAML
        try:
            yaml_content = yaml.safe_load(config_yaml)
            if not isinstance(yaml_content, dict):
                errors.append("config.yml must be a valid YAML object")
            elif "models" not in yaml_content:
                errors.append("config.yml missing required 'models' section")
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML syntax: {str(e)}")

        # Basic Colang validation (just check it's not empty for now)
        if not config_colang.strip():
            errors.append("config.co cannot be empty")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    except Exception as e:
        logger.error("Error validating content: %s", e)
        return {"valid": False, "errors": [str(e)]}


@router.post("/{agent_name}/test")
async def test_agent(agent_name: str, test_data: dict):
    """
    Test an agent with a sample input.

    Args:
        agent_name: Name of the agent to test
        test_data: Dict with input text

    Returns:
        Test result with output and triggered rails.
    """
    try:
        import time
        start_time = time.time()

        input_text = test_data.get("input", "").strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="Input is required")

        # For now, return a mock result
        # In a real implementation, this would run the guardrails
        execution_time = int((time.time() - start_time) * 1000)

        return {
            "blocked": False,
            "modified": False,
            "output": f"Test response for: {input_text}",
            "triggered_rails": [],
            "execution_time": execution_time,
            "details": {
                "agent": agent_name,
                "input": input_text,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error testing agent '%s': %s", agent_name, e)
        raise HTTPException(status_code=500, detail=str(e)) from e
