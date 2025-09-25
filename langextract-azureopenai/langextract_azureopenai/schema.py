"""Schema implementation for AzureOpenAI provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import langextract as lx  # type: ignore[import-untyped]


class AzureOpenAISchema(lx.core.schema.BaseSchema):
    """Schema implementation for AzureOpenAI structured output.
    
    This schema is designed to work with GPT-5's new structured outputs feature,
    which uses the response_format with json_schema type for strict JSON validation.
    """

    def __init__(self, schema_dict: dict[str, Any]) -> None:
        """Initialize the schema with a dictionary."""
        self._schema_dict = schema_dict

    @property
    def schema_dict(self) -> dict[str, Any]:
        """Return the schema dictionary."""
        return self._schema_dict

    @classmethod
    def from_examples(
        cls, examples_data: Sequence[Any], attribute_suffix: str = "_attributes"
    ) -> AzureOpenAISchema:
        """Build schema from example extractions.

        Args:
            examples_data: Sequence of ExampleData objects.
            attribute_suffix: Suffix for attribute fields.

        Returns:
            A configured AzureOpenAISchema instance.
        """
        extraction_types: dict[str, set[str]] = {}
        for example in examples_data:
            for extraction in example.extractions:
                class_name = extraction.extraction_class
                if class_name not in extraction_types:
                    extraction_types[class_name] = set()
                if extraction.attributes:
                    extraction_types[class_name].update(extraction.attributes.keys())

        # Build a detailed schema for GPT-5 structured outputs that matches LangExtract's format
        # This schema ensures that the model returns properly structured extractions
        extraction_item_properties = {
            "extraction_class": {
                "type": "string",
                "description": "The class or type of this extraction (e.g., 'character', 'emotion', 'relationship')",
                "enum": list(extraction_types.keys()) if extraction_types else ["character", "emotion", "relationship"]
            },
            "extraction_text": {
                "type": "string", 
                "description": "The exact text span extracted from the source document"
            },
            "attributes": {
                "type": "object",
                "description": "Additional metadata and context for this extraction",
                "properties": {},
                "additionalProperties": {
                    "type": "string"
                }
            }
        }

        # Add specific attribute schemas based on examples
        for class_name, attrs in extraction_types.items():
            if attrs:
                class_description = f"Attributes specific to {class_name} extractions"
                extraction_item_properties["attributes"]["description"] = class_description

        schema_dict: dict[str, Any] = {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "description": "Array of extracted entities from the text",
                    "items": {
                        "type": "object",
                        "properties": extraction_item_properties,
                        "required": ["extraction_class", "extraction_text", "attributes"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["extractions"],
            "additionalProperties": False
        }

        return cls(schema_dict)

    def to_provider_config(self) -> dict[str, Any]:
        """Convert to provider-specific configuration.

        Returns:
            Dictionary of provider-specific configuration.
        """
        # Map to provider kwargs. We expose enable_structured_output to
        # align with provider plugin guidelines.
        return {
            "response_schema": self._schema_dict,
            "enable_structured_output": True,
        }

    @property
    def requires_raw_output(self) -> bool:
        """Whether this schema requires raw JSON output without fences.

        Returns:
            True since Azure OpenAI GPT-5 structured outputs returns clean JSON.
        """
        return True

    @property
    def supports_strict_mode(self) -> bool:
        """Whether this schema guarantees valid structured output.

        Returns:
            True if the provider enforces valid JSON output (no code fences).
        """
        # When this schema is applied, the provider enables GPT-5's structured outputs
        # which returns well-formed JSON without markdown fencing.
        return True

    def validate_format(self, format_handler: Any) -> None:
        """Validate Azure OpenAI format requirements.

        Azure OpenAI GPT-5 structured outputs requires:
        - No fence markers (outputs raw JSON via json_schema response_format)
        - Wrapper with EXTRACTIONS_KEY (built into response_schema)
        """
        # Check for fence usage with raw JSON output
        if format_handler.use_fences:
            import warnings
            warnings.warn(
                "Azure OpenAI GPT-5 outputs native JSON via"
                " structured outputs. Using fence_output=True may"
                " cause parsing issues. Set fence_output=False.",
                UserWarning,
                stacklevel=3,
            )

        # Verify wrapper is enabled with correct key
        if (
            not format_handler.use_wrapper
            or format_handler.wrapper_key != "extractions"
        ):
            import warnings
            warnings.warn(
                "Azure OpenAI's response_schema expects"
                f" wrapper_key='extractions'. Current settings:"
                f" use_wrapper={format_handler.use_wrapper},"
                f" wrapper_key='{format_handler.wrapper_key}'",
                UserWarning,
                stacklevel=3,
            )
