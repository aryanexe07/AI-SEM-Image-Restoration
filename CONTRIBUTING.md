# Contributing Guidelines

Thank you for contributing to the **AI-Based Restoration of Degraded SEM Images using NAFNet** project.

## Development Workflow

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/harshwardhan1507/AI-SEM-Image-Restoration.git
   cd AI-SEM-Image-Restoration
   ```

2. **Set Up Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Install Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

## Documentation & Code Standards

- **Code Style**: Follow PEP8. Format code using `black` and lint with `ruff`.
- **Imports**: Group and sort imports using `isort`. No wildcard imports allowed (`from module import *`). Always use absolute imports rooted in `src`.
- **Type Hints**: All functions, methods, and classes must include full type hints.
- **Docstrings**: Write Google-style docstrings for all public modules, functions, and classes.
- **Testing**: Add pytest test cases under `tests/` for new functionality.

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/your-feature-name`).
2. Run linters and tests before committing:
   ```bash
   ruff check .
   black --check .
   pytest
   ```
3. Commit changes with clear, descriptive commit messages.
4. Push your branch and open a Pull Request.
