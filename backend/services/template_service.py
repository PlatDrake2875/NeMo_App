"""
Template Service - Manages experiment configuration templates.

Provides operations for saving, loading, listing, and managing YAML-based
experiment templates for reproducible preprocessing and evaluation configurations.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from schemas.experiment_template import (
    ExperimentTemplate,
    TemplateMetadata,
    TemplateListResponse,
)

logger = logging.getLogger(__name__)


class TemplateService:
    """Service for managing experiment configuration templates."""

    # Default directories
    TEMPLATES_DIR = Path("data/experiment_templates")
    PRESETS_DIR = Path("data/experiment_templates/presets")

    def __init__(self, templates_dir: Optional[Path] = None, presets_dir: Optional[Path] = None):
        """
        Initialize the template service.

        Args:
            templates_dir: Custom directory for user templates
            presets_dir: Custom directory for built-in presets
        """
        self.templates_dir = templates_dir or self.TEMPLATES_DIR
        self.presets_dir = presets_dir or self.PRESETS_DIR

        # Ensure directories exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a template name for use as a filename."""
        # Replace spaces and special chars with underscores
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        # Remove consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_").lower()

    def _get_template_path(self, name: str, is_preset: bool = False) -> Path:
        """Get the file path for a template."""
        base_dir = self.presets_dir if is_preset else self.templates_dir
        filename = f"{self._sanitize_filename(name)}.yaml"
        return base_dir / filename

    def save_template(
        self,
        template: ExperimentTemplate,
        overwrite: bool = False,
    ) -> Path:
        """
        Save a template to the templates directory.

        Args:
            template: The template to save
            overwrite: Whether to overwrite existing template

        Returns:
            Path to the saved template file

        Raises:
            FileExistsError: If template exists and overwrite is False
        """
        file_path = self._get_template_path(template.name)

        if file_path.exists() and not overwrite:
            raise FileExistsError(
                f"Template '{template.name}' already exists. Use overwrite=True to replace."
            )

        # Update timestamps
        now = datetime.now(timezone.utc).isoformat()
        if not template.created_at:
            template.created_at = now
        template.updated_at = now

        # Write to file
        yaml_content = template.to_yaml()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        logger.info(f"Saved template '{template.name}' to {file_path}")
        return file_path

    def load_template(self, name: str) -> ExperimentTemplate:
        """
        Load a template by name.

        First checks user templates, then presets.

        Args:
            name: Name of the template to load

        Returns:
            The loaded ExperimentTemplate

        Raises:
            FileNotFoundError: If template not found
        """
        # Check user templates first
        user_path = self._get_template_path(name, is_preset=False)
        if user_path.exists():
            return ExperimentTemplate.from_yaml_file(str(user_path))

        # Check presets
        preset_path = self._get_template_path(name, is_preset=True)
        if preset_path.exists():
            return ExperimentTemplate.from_yaml_file(str(preset_path))

        # Try exact filename match in both directories
        for directory in [self.templates_dir, self.presets_dir]:
            for yaml_file in directory.glob("*.yaml"):
                try:
                    template = ExperimentTemplate.from_yaml_file(str(yaml_file))
                    if template.name.lower() == name.lower():
                        return template
                except Exception:
                    continue

        raise FileNotFoundError(f"Template '{name}' not found")

    def delete_template(self, name: str) -> bool:
        """
        Delete a user template.

        Note: Cannot delete built-in presets.

        Args:
            name: Name of the template to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_template_path(name, is_preset=False)

        if not file_path.exists():
            return False

        file_path.unlink()
        logger.info(f"Deleted template '{name}'")
        return True

    def list_templates(
        self,
        include_presets: bool = True,
        tags: Optional[List[str]] = None,
    ) -> TemplateListResponse:
        """
        List all available templates.

        Args:
            include_presets: Whether to include built-in presets
            tags: Filter by tags (any match)

        Returns:
            TemplateListResponse with list of template metadata
        """
        templates: List[TemplateMetadata] = []

        # List user templates
        for yaml_file in self.templates_dir.glob("*.yaml"):
            try:
                template = ExperimentTemplate.from_yaml_file(str(yaml_file))
                metadata = TemplateMetadata(
                    name=template.name,
                    description=template.description,
                    version=template.version,
                    tags=template.tags,
                    author=template.author,
                    created_at=template.created_at,
                    updated_at=template.updated_at,
                    content_hash=template.content_hash(),
                    file_path=str(yaml_file),
                    is_builtin=False,
                )
                templates.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load template {yaml_file}: {e}")

        # List presets if requested
        if include_presets:
            for yaml_file in self.presets_dir.glob("*.yaml"):
                try:
                    template = ExperimentTemplate.from_yaml_file(str(yaml_file))
                    metadata = TemplateMetadata(
                        name=template.name,
                        description=template.description,
                        version=template.version,
                        tags=template.tags,
                        author=template.author,
                        created_at=template.created_at,
                        updated_at=template.updated_at,
                        content_hash=template.content_hash(),
                        file_path=str(yaml_file),
                        is_builtin=True,
                    )
                    templates.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to load preset {yaml_file}: {e}")

        # Filter by tags if specified
        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]

        # Sort by name
        templates.sort(key=lambda t: (not t.is_builtin, t.name.lower()))

        return TemplateListResponse(
            templates=templates,
            total=len(templates),
        )

    def get_builtin_presets(self) -> List[ExperimentTemplate]:
        """
        Get all built-in preset templates.

        Returns:
            List of built-in ExperimentTemplate objects
        """
        presets: List[ExperimentTemplate] = []

        for yaml_file in self.presets_dir.glob("*.yaml"):
            try:
                template = ExperimentTemplate.from_yaml_file(str(yaml_file))
                presets.append(template)
            except Exception as e:
                logger.warning(f"Failed to load preset {yaml_file}: {e}")

        return presets

    def template_exists(self, name: str) -> bool:
        """Check if a template exists (user or preset)."""
        user_path = self._get_template_path(name, is_preset=False)
        preset_path = self._get_template_path(name, is_preset=True)
        return user_path.exists() or preset_path.exists()

    def export_template(self, name: str) -> str:
        """
        Export a template as YAML string (for download).

        Args:
            name: Name of the template to export

        Returns:
            YAML string content
        """
        template = self.load_template(name)
        return template.to_yaml()

    def import_template(
        self,
        yaml_content: str,
        name_override: Optional[str] = None,
        overwrite: bool = False,
    ) -> ExperimentTemplate:
        """
        Import a template from YAML content.

        Args:
            yaml_content: YAML string content
            name_override: Optional name to use instead of the one in YAML
            overwrite: Whether to overwrite existing template

        Returns:
            The imported ExperimentTemplate
        """
        template = ExperimentTemplate.from_yaml(yaml_content)

        if name_override:
            template.name = name_override

        self.save_template(template, overwrite=overwrite)
        return template


# Singleton instance
template_service = TemplateService()
