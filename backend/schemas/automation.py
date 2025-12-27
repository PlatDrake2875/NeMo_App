"""Schemas for automation/workflow endpoints."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.common import Message


class AutomateRequest(BaseModel):
    """Request for automating conversation analysis and task generation."""

    conversation_history: list[Message] = Field(
        ..., description="The current conversation history to be automated."
    )
    model: str = Field(..., description="The model to use for the automation task.")
    automation_task: Optional[str] = Field(
        None,
        description="Specific automation task to perform (e.g., 'summarize', 'generate_next_steps').",
    )
    config_params: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional configuration parameters for automation.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "We discussed the project timeline and deliverables.",
                    },
                    {
                        "role": "assistant",
                        "content": "Okay, I've noted that. The key deliverables are X, Y, and Z due by next Friday.",
                    },
                    {
                        "role": "user",
                        "content": "Correct. Also, remember to schedule the follow-up meeting.",
                    },
                ],
                "model": "llama3:latest",
                "automation_task": "generate_meeting_summary_and_actions",
                "config_params": {"max_summary_length": 200},
            }
        }
    )


class AutomateResponse(BaseModel):
    """Response from an automation operation."""

    status: str = Field(
        ..., description="Status of the automation request (e.g., 'success', 'error')."
    )
    message: Optional[str] = Field(
        None, description="A message providing details about the outcome."
    )
    data: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Output data from the automation process."
    )
    error_details: Optional[str] = Field(
        None, description="Details if an error occurred."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Conversation automated successfully.",
                "data": {
                    "summary": "The project timeline and deliverables were discussed. Key items are X, Y, Z due next Friday. A follow-up meeting needs to be scheduled.",
                    "action_items": ["Schedule follow-up meeting."],
                },
            }
        }
    )
