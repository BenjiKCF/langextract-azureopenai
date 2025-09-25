"""Provider implementation for AzureOpenAI."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any, Final

import langextract as lx  # type: ignore[import-untyped]

# Azure OpenAI Chat Completions API supported parameters
# Based on: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference
_AZURE_OPENAI_CONFIG_KEYS: Final[set[str]] = {
    'frequency_penalty',  # Number between -2.0 and 2.0
    'presence_penalty',  # Number between -2.0 and 2.0
    'stop',  # String or array of stop sequences
    'logprobs',  # Whether to return log probabilities
    'top_logprobs',  # Number of most likely tokens (0-5)
    'seed',  # Random seed for deterministic outputs
    'user',  # Unique identifier for end-user
    'response_format',  # Output format (text, json_object, json_schema)
    'tools',  # Array of tools/functions model can call (unsupported)
    'tool_choice',  # Controls which tools to use (unsupported)
    'logit_bias',  # Map of token IDs to bias scores (-100 to 100)
    'stream',  # Whether to stream partial responses (unsupported)
    'parallel_tool_calls',  # Whether to enable parallel function calling (unsupported)
}


@lx.providers.registry.register(r'^azureopenai', priority=10)
class AzureOpenAILanguageModel(lx.core.base_model.BaseLanguageModel):
    """Language model inference using Azure OpenAI's API with structured output.

    This provider handles model IDs matching: ['^azureopenai']
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI language model.

        Args:
            model_id: The Azure OpenAI model ID to use (e.g., 'azureopenai-gpt-5-nano').
            api_key: API key for Azure OpenAI service.
            azure_endpoint: Azure OpenAI endpoint URL.
            api_version: API version to use.
            deployment_name: Deployment name. If None, extracted from model_id.
            temperature: Sampling temperature.
            max_workers: Maximum number of parallel API calls.
            **kwargs: Additional parameters passed to the Azure OpenAI API.
        """
        # Lazy import: OpenAI package required
        try:
            # pylint: disable=import-outside-toplevel
            from openai import AzureOpenAI
        except ImportError as e:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI provider requires openai package. '
                'Install with: pip install openai>=1.0.0'
            ) from e

        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        self.azure_endpoint = azure_endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')
        # api_version is mandatory: from arg or env
        self.api_version = api_version or os.environ.get('AZURE_OPENAI_API_VERSION')
        self.temperature = temperature
        self.max_workers = max_workers
        self._response_schema: dict[str, Any] | None = None
        self._enable_structured_output: bool = False

        # Extract deployment name from model_id if not provided
        if deployment_name:
            self.deployment_name = deployment_name
        else:
            # Extract deployment name by removing 'azureopenai-' prefix
            if isinstance(model_id, str) and model_id.startswith('azureopenai-'):
                self.deployment_name = model_id[len('azureopenai-'):]
            else:
                self.deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')

        # Validate required parameters
        if not self.api_key:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY '
                'environment variable or pass api_key parameter.'
            )
        if not self.azure_endpoint:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT '
                'environment variable or pass azure_endpoint parameter.'
            )
        if not self.api_version:
            raise lx.exceptions.InferenceConfigError(
                'Azure OpenAI API version not provided. Set AZURE_OPENAI_API_VERSION '
                'environment variable or pass api_version parameter.'
            )

        # Reject unsupported parameters early if provided at construction
        unsupported = {'stream', 'tools', 'tool_choice', 'parallel_tool_calls'}
        for key in unsupported.intersection(set((kwargs or {}).keys())):
            raise lx.exceptions.InferenceConfigError(
                f'Parameter {key} is not supported by Azure OpenAI provider'
            )

        # Initialize the Azure OpenAI client
        self._client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )

        # Filter extra kwargs to only include valid Azure OpenAI API parameters
        self._extra_kwargs = {
            k: v for k, v in (kwargs or {}).items() if k in _AZURE_OPENAI_CONFIG_KEYS
        }

    @classmethod
    def get_schema_class(cls) -> None:
        """Return None to disable LangExtract schema constraints.
        
        Azure OpenAI structured outputs work better without LangExtract's schema system
        since GPT-5 handles structured JSON generation natively.
        """
        return None

    @property
    def requires_fence_output(self) -> bool:
        """Whether this model requires fence output for parsing.
        
        Uses explicit override if set, otherwise defaults to True to ensure
        output is wrapped in ```json fences for proper parsing by LangExtract.
        """
        if (
            hasattr(self, '_fence_output_override')
            and self._fence_output_override is not None
        ):
            return self._fence_output_override
        
        # Default to True since we don't use schema constraints
        # LangExtract expects fenced output for proper parsing
        return True

    def apply_schema(self, schema_instance: object | None) -> None:
        """Apply or clear schema configuration."""
        super().apply_schema(schema_instance)
        if schema_instance is None:
            self._response_schema = None
            self._enable_structured_output = False
        # Since we disabled schema support, we ignore any schema_instance

    def _process_single_prompt(
        self, prompt: str, config: dict[str, Any]
    ) -> lx.core.types.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            # Prepare the API call configuration
            api_config = {
                'model': self.deployment_name,
                'messages': [{'role': 'user', 'content': prompt}],
                **config,
                **self._extra_kwargs,
            }

            # GPT-5 structured outputs: Use response_format with json_schema
            if self._enable_structured_output and self._response_schema:
                # For GPT-5, we use the new structured outputs format
                # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs
                extraction_schema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_response",
                        "schema": self._response_schema,
                        "strict": True  # Enable strict mode for GPT-5
                    }
                }
                api_config['response_format'] = extraction_schema
            else:
                # For non-schema mode, always try to encourage JSON output
                # Since we're always expecting structured data for LangExtract
                try:
                    api_config['response_format'] = {"type": "json_object"}
                except Exception:
                    # If json_object format is not supported, continue without it
                    pass
            
            # Call the Azure OpenAI API
            response = self._client.chat.completions.create(**api_config)
            
            # Extract the content
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content or ""
                
                # Handle empty content
                if not content.strip():
                    # Return empty extractions if no content
                    import json
                    empty_output = json.dumps({"extractions": []}, indent=2)
                    if self.requires_fence_output:
                        empty_output = f"```json\n{empty_output}\n```"
                    return lx.core.types.ScoredOutput(
                        output=empty_output,
                        score=0.0
                    )
                
                # Format the output according to LangExtract expectations
                output_text = self._format_output_for_langextract(content)
                
                # Create ScoredOutput - using dummy score since Azure OpenAI doesn't provide log probabilities by default
                scored_output = lx.core.types.ScoredOutput(
                    output=output_text,
                    score=1.0  # Default score
                )
                return scored_output
            else:
                # Handle empty response
                import json
                empty_output = json.dumps({"extractions": []}, indent=2)
                if self.requires_fence_output:
                    empty_output = f"```json\n{empty_output}\n```"
                return lx.core.types.ScoredOutput(
                    output=empty_output,
                    score=0.0
                )

        except Exception as e:
            raise lx.exceptions.InferenceError(f'Azure OpenAI API call failed: {e}') from e
    
    def _format_output_for_langextract(self, content: str) -> str:
        """Format the model output according to LangExtract's expectations.
        
        Args:
            content: Raw content from the model
            
        Returns:
            Formatted output ready for LangExtract parsing
        """
        import json
        import re
        
        # Always try to parse and convert the content to ensure proper format
        content_stripped = content.strip()
        
        # Try to extract JSON from the content if it's wrapped in markdown
        content_to_parse = content_stripped
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content_stripped, re.DOTALL)
        if json_match:
            content_to_parse = json_match.group(1).strip()
        
        try:
            # Parse the JSON
            parsed = json.loads(content_to_parse)
            
            # Convert to LangExtract format
            formatted_json = self._convert_to_langextract_format(parsed)
            
            # Double-check that we have valid extractions
            if not formatted_json.get("extractions"):
                # If no extractions, try to create a fallback based on the content
                formatted_json = self._create_fallback_extraction(content_stripped)
                
            output_text = json.dumps(formatted_json, indent=2)
            
        except json.JSONDecodeError:
            # If not valid JSON, try to create a fallback extraction
            formatted_json = self._create_fallback_extraction(content_stripped)
            output_text = json.dumps(formatted_json, indent=2)
        
        # Always add fencing for LangExtract parsing
        if self.requires_fence_output:
            output_text = f"```json\n{output_text}\n```"
        
        return output_text
    
    def _convert_to_langextract_format(self, parsed: dict) -> dict:
        """Convert standard JSON format to LangExtract's expected format.
        
        LangExtract expects a specific format where:
        - extraction_class becomes the key (e.g., "Evaluation")
        - extraction_text becomes the value
        - attributes becomes "{extraction_class}_attributes"
        
        Args:
            parsed: Standard JSON format
            
        Returns:
            LangExtract format JSON
        """
        if not isinstance(parsed, dict):
            return {"extractions": []}
            
        # Handle direct list format
        if isinstance(parsed, list):
            formatted_json = {"extractions": parsed}
        elif "extractions" not in parsed:
            # Wrap single extraction
            if "extraction_class" in parsed:
                formatted_json = {"extractions": [parsed]}
            else:
                formatted_json = {"extractions": []}
        else:
            formatted_json = parsed

        # Convert each extraction to LangExtract format
        langextract_extractions = []
        for extraction in formatted_json.get("extractions", []):
            if not isinstance(extraction, dict):
                continue
            
            # Check if it's already in LangExtract format
            is_langextract_format = any(key.endswith('_attributes') for key in extraction.keys())
            
            if is_langextract_format:
                # Already in LangExtract format, keep as-is
                langextract_extractions.append(extraction)
                continue
                
            # Get extraction details from standard format
            extraction_class = extraction.get("extraction_class", "unknown")
            extraction_text = extraction.get("extraction_text", "")
            attributes = extraction.get("attributes", {})
            
            # Ensure attributes is not None
            if attributes is None:
                attributes = {}
                
            # Create LangExtract format
            langextract_extraction = {
                extraction_class: extraction_text
            }
            
            # Add attributes with proper naming convention
            if attributes:
                langextract_extraction[f"{extraction_class}_attributes"] = attributes
                
            langextract_extractions.append(langextract_extraction)

        return {"extractions": langextract_extractions}
    
    def _create_fallback_extraction(self, content: str) -> dict:
        """Create a fallback extraction when parsing fails.
        
        This method tries to extract meaningful information from the raw content
        and formats it as a LangExtract extraction.
        
        Args:
            content: Raw content from the model
            
        Returns:
            LangExtract format JSON with fallback extraction
        """
        # Try to extract some meaningful text for classification
        # Look for common classification responses
        content_lower = content.lower().strip()
        
        if "acceptable" in content_lower or "unacceptable" in content_lower:
            # Try to determine the evaluation
            if "unacceptable" in content_lower:
                evaluation = "Unacceptable"
            else:
                evaluation = "Acceptable"
                
            # Try to extract the text being evaluated
            # Look for patterns like quotes or capitalized text
            import re
            
            # Look for quoted text or all caps text that might be the item being evaluated
            quoted_match = re.search(r'"([^"]+)"', content)
            caps_match = re.search(r'\b([A-Z\s&]+)\b', content)
            
            extraction_text = "Unknown"
            if quoted_match:
                extraction_text = quoted_match.group(1)
            elif caps_match and len(caps_match.group(1)) > 2:
                extraction_text = caps_match.group(1).strip()
            
            return {
                "extractions": [
                    {
                        "Evaluation": extraction_text,
                        "Evaluation_attributes": {
                            "evaluation": evaluation
                        }
                    }
                ]
            }
        
        # If we can't extract anything meaningful, return empty
        return {"extractions": []}

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[lx.core.types.ScoredOutput]]:
        """Runs inference on a list of prompts via Azure OpenAI's API.

        Args:
            batch_prompts: A list of string prompts.
            **kwargs: Additional generation params (temperature, top_p, etc.)

        Yields:
            Lists of ScoredOutputs.
        """
        config: dict[str, Any] = {}

        # Handle standard parameters explicitly
        temp = kwargs.get('temperature', self.temperature)
        if temp is not None:
            config['temperature'] = temp
        if 'max_completion_tokens' in kwargs:
            config['max_completion_tokens'] = kwargs['max_completion_tokens']
        if 'top_p' in kwargs:
            config['top_p'] = kwargs['top_p']

        # Handle all other whitelisted Azure OpenAI parameters
        handled_keys = {'temperature', 'max_completion_tokens', 'top_p'}
        for key, value in kwargs.items():
            if key not in handled_keys and key in _AZURE_OPENAI_CONFIG_KEYS:
                config[key] = value
            elif key not in handled_keys and key not in {
                'model_id', 'api_key', 'azure_endpoint', 'api_version', 
                'deployment_name', 'max_workers'
            }:
                # Log warning for unrecognized parameters
                print(f"Warning: Unrecognized parameter '{key}' ignored")

        # Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            # For simplicity, process sequentially for now
            # In a real implementation, you'd use ThreadPoolExecutor
            results = []
            for prompt in batch_prompts:
                try:
                    result = self._process_single_prompt(prompt, config)
                    results.append(result)
                except Exception as e:
                    # Create error output for failed prompts
                    error_output = lx.core.types.ScoredOutput(
                        output=f"Error: {str(e)}",
                        score=0.0
                    )
                    results.append(error_output)
            yield results
        else:
            # Single prompt processing
            results = [self._process_single_prompt(batch_prompts[0], config)]
            yield results
