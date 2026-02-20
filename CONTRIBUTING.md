# Contributing to Chains

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/your-username/chains.git
cd chains
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/ -v
```

## Code Style

We use `ruff` for linting:

```bash
ruff check chains/ tests/
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a PR with a clear description
