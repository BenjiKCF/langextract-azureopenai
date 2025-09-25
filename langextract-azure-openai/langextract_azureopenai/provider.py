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
                return lx.core.types.ScoredOutput(
                    output="",
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
        # For structured outputs, the content should already be valid JSON
        if self._enable_structured_output:
            # GPT-5 structured outputs return clean JSON without markdown fences
            output_text = content.strip()
            # Check if we need to add fencing
            if self.requires_fence_output:
                # Wrap in markdown fences for LangExtract parsing
                output_text = f"```json\n{output_text}\n```"
            return output_text
        
        # For non-structured output, check if the model already returned properly formatted content
        content_stripped = content.strip()
        
        # If content is already fenced and we expect fences, return as-is
        if self.requires_fence_output and content_stripped.startswith('```') and content_stripped.endswith('```'):
            return content_stripped
        
        # If content is not fenced but we don't expect fences, and it's valid JSON, return as-is
        if not self.requires_fence_output:
            import json
            try:
                # Try to parse as JSON to validate
                json.loads(content_stripped)
                return content_stripped
            except json.JSONDecodeError:
                pass
        
        # If we get here, we need to format the content properly
        import json
        import re
        
        # Try to extract JSON from the content if it's wrapped in markdown
        content_to_parse = content_stripped
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content_stripped, re.DOTALL)
        if json_match:
            content_to_parse = json_match.group(1).strip()
        
        try:
            # Check if content is valid JSON
            parsed = json.loads(content_to_parse)
            
            # Ensure it has the extractions structure
            if not isinstance(parsed, dict) or "extractions" not in parsed:
                # Wrap in extractions structure
                if isinstance(parsed, list):
                    formatted_json = {"extractions": parsed}
                else:
                    formatted_json = {"extractions": [parsed]}
            else:
                formatted_json = parsed
                
            output_text = json.dumps(formatted_json, indent=2)
            
        except json.JSONDecodeError:
            # If not valid JSON, return an empty extractions array
            formatted_json = {
                "extractions": []
            }
            output_text = json.dumps(formatted_json, indent=2)
        
        # Check if we need to add fencing
        if self.requires_fence_output:
            # Wrap in markdown fences for LangExtract parsing
            output_text = f"```json\n{output_text}\n```"
        
        return output_text

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
