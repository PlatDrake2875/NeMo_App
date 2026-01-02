"""
Message conversion utilities.
Handles conversion between different message formats used across the application.
"""

from typing import Optional


# Role mapping constants
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_BOT = "bot"


class MessageConverter:
    """Utility class for converting message formats."""

    @staticmethod
    def history_to_llm_messages(
        history: list[dict],
        current_query: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Convert chat history to LLM-compatible message format.

        Args:
            history: List of message dicts with 'sender' and 'text' keys
            current_query: Optional current user query to append

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages: list[dict[str, str]] = []

        for msg in history:
            role = msg.get("sender", ROLE_USER).lower()
            content = msg.get("text", "")

            # Convert 'bot' role to 'assistant' for OpenAI compatibility
            if role == ROLE_BOT:
                messages.append({"role": ROLE_ASSISTANT, "content": content})
            else:
                messages.append({"role": ROLE_USER, "content": content})

        if current_query is not None:
            messages.append({"role": ROLE_USER, "content": current_query})

        return messages

    @staticmethod
    def extract_last_user_message(messages: list[dict[str, str]]) -> str:
        """
        Extract the last user message from a message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            The content of the last user message

        Raises:
            ValueError: If no user message is found
        """
        for msg in reversed(messages):
            if msg.get("role") == ROLE_USER:
                content = msg.get("content", "")
                if content:
                    return content

        raise ValueError("No user message found in messages")

    @staticmethod
    def normalize_role(role: str) -> str:
        """
        Normalize a role string to standard format.

        Args:
            role: Role string (e.g., 'bot', 'assistant', 'user')

        Returns:
            Normalized role ('user' or 'assistant')
        """
        role_lower = role.lower()
        if role_lower in (ROLE_BOT, ROLE_ASSISTANT):
            return ROLE_ASSISTANT
        return ROLE_USER
