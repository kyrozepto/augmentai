# Contributing to AugmentAI

Thank you for your interest in contributing to AugmentAI! 🎨

## Getting Started

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/kyrozepto/aai.git
   cd aai
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=augmentai --cov-report=html
```

### Code Style
We use the following tools for code quality:
- **Black** for formatting
- **Ruff** for linting
- **MyPy** for type checking

Run them before submitting:
```bash
black augmentai/
ruff check augmentai/
mypy augmentai/
```

## Submitting Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure tests pass

3. **Write tests** for new functionality

4. **Commit with a clear message**:
   ```bash
   git commit -m "feat: add support for new transform type"
   ```

5. **Push and create a Pull Request**

## Adding a New Domain

To add a new domain (e.g., `dental`):

1. Create `augmentai/domains/dental.py`:
   ```python
   from augmentai.domains.base import Domain, DomainConstraint, ConstraintLevel

   class DentalDomain(Domain):
       name = "dental"
       description = "Dental X-ray imaging with tooth preservation constraints"
       
       def _setup(self) -> None:
           # Add your domain-specific constraints
           self.add_constraint(DomainConstraint(
               transform_name="ElasticTransform",
               level=ConstraintLevel.FORBIDDEN,
               reason="Can distort tooth structures"
           ))
           
           self.recommended_transforms.add("HorizontalFlip")
   ```

2. Register in `augmentai/domains/__init__.py`

3. Add tests in `tests/test_domains.py`

## Adding a New Transform

1. Add the transform spec to `augmentai/core/schema.py`:
   ```python
   schema.register(TransformSpec(
       name="YourTransform",
       category=TransformCategory.COLOR,
       description="Your transform description",
       parameters={
           "param1": ParameterRange("param1", 0.0, 1.0, 0.5, "float"),
       },
   ))
   ```

2. Add mapping in `augmentai/compilers/albumentations.py`

3. Add tests

## Reporting Issues

- Use the issue template when available
- Include your Python version and OS
- Provide a minimal reproducible example
- Include the full error traceback

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers warmly
- Focus on constructive feedback

## Questions?

Feel free to open a discussion or issue if you have questions!

---

Thank you for contributing! 🙏
