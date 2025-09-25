        # LangExtract AzureOpenAI Provider

A provider plugin for LangExtract that supports AzureOpenAI models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

- `azureopenai*`: Models matching pattern ^azureopenai

## Environment Variables

- `AZUREOPENAI_API_KEY`: API key for authentication
- `AZURE_OPENAI_ENDPOINT`: endpoint for authentication
- `AZURE_OPENAI_API_VERSION`: API version for authentication
- `AZURE_OPENAI_DEPLOYMENT_NAME`: deployment name for authentication

## Usage

```python
import langextract as lx

# Run the extraction using Azure OpenAI GPT-5 provider
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id=f"azureopenai-{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}",  # Use our Azure OpenAI provider
    debug=False
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
