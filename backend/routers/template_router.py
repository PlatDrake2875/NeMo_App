"""
Template Router - API endpoints for experiment template management.

Provides REST API for:
- Listing templates and presets
- Saving and loading templates
- Importing and exporting YAML files
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse

from schemas.experiment_template import (
    ExperimentTemplate,
    TemplateMetadata,
    TemplateListResponse,
    SaveTemplateRequest,
    SaveTemplateResponse,
    ImportTemplateRequest,
    ExportTemplateResponse,
)
from services.template_service import template_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/templates", tags=["templates"])


@router.get("/presets", response_model=List[TemplateMetadata])
async def list_presets():
    """
    List all built-in preset templates.

    Returns only the built-in presets, not user-saved templates.
    """
    try:
        response = template_service.list_templates(include_presets=True)
        return [t for t in response.templates if t.is_builtin]
    except Exception as e:
        logger.error(f"Failed to list presets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets/{name}", response_model=ExperimentTemplate)
async def get_preset(name: str):
    """
    Get a specific preset by name.

    Args:
        name: Name of the preset to retrieve
    """
    try:
        template = template_service.load_template(name)
        return template
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to load preset '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=TemplateListResponse)
async def list_templates(
    include_presets: bool = Query(True, description="Include built-in presets"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
):
    """
    List all available templates.

    Args:
        include_presets: Whether to include built-in presets in the response
        tags: Comma-separated list of tags to filter by
    """
    try:
        tag_list = tags.split(",") if tags else None
        return template_service.list_templates(
            include_presets=include_presets,
            tags=tag_list,
        )
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=SaveTemplateResponse)
async def save_template(request: SaveTemplateRequest):
    """
    Save a new template or update an existing one.

    Args:
        request: SaveTemplateRequest with the template to save
    """
    try:
        # Check if template exists
        exists = template_service.template_exists(request.template.name)

        if exists and not request.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"Template '{request.template.name}' already exists. Set overwrite=true to replace.",
            )

        file_path = template_service.save_template(
            request.template,
            overwrite=request.overwrite,
        )

        return SaveTemplateResponse(
            name=request.template.name,
            file_path=str(file_path),
            content_hash=request.template.content_hash(),
            created=not exists,
        )
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}", response_model=ExperimentTemplate)
async def get_template(name: str):
    """
    Get a template by name.

    Searches user templates first, then presets.

    Args:
        name: Name of the template to retrieve
    """
    try:
        return template_service.load_template(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to load template '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}")
async def delete_template(name: str):
    """
    Delete a user template.

    Note: Cannot delete built-in presets.

    Args:
        name: Name of the template to delete
    """
    try:
        # Check if it's a preset
        response = template_service.list_templates(include_presets=True)
        for t in response.templates:
            if t.name.lower() == name.lower() and t.is_builtin:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot delete built-in presets",
                )

        deleted = template_service.delete_template(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")

        return {"message": f"Template '{name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=ExportTemplateResponse)
async def export_template(name: str = Query(..., description="Template name to export")):
    """
    Export a template as YAML for download.

    Args:
        name: Name of the template to export
    """
    try:
        yaml_content = template_service.export_template(name)
        filename = f"{name.lower().replace(' ', '_')}.yaml"

        return ExportTemplateResponse(
            yaml_content=yaml_content,
            filename=filename,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to export template '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{name}/download")
async def download_template(name: str):
    """
    Download a template as a YAML file.

    Args:
        name: Name of the template to download
    """
    try:
        yaml_content = template_service.export_template(name)
        filename = f"{name.lower().replace(' ', '_')}.yaml"

        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to download template '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import", response_model=ExperimentTemplate)
async def import_template(request: ImportTemplateRequest):
    """
    Import a template from YAML content.

    Args:
        request: ImportTemplateRequest with YAML content
    """
    try:
        template = template_service.import_template(
            yaml_content=request.yaml_content,
            name_override=request.name,
            overwrite=False,
        )
        return template
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to import template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
