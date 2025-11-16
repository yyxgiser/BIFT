# Contributing to BIFT

Thank you for your interest in contributing to BIFT! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/BIFT.git
   cd BIFT
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards below

3. Test your changes:
   ```bash
   python test_bift.py
   ```

4. Commit your changes with a clear message:
   ```bash
   git commit -m "Add: description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request on GitHub

## Coding Standards

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Documentation

- Update the README if you add new features
- Add docstrings following NumPy/Google style
- Include usage examples for new functionality
- Update the examples directory if appropriate

## Testing

- Write tests for new functionality
- Ensure all existing tests pass
- Test with different parameter configurations
- Verify that examples still work

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version
- OpenCV version
- Operating system
- Minimal code to reproduce the issue
- Expected vs actual behavior

### Feature Requests

When requesting features, please:
- Explain the use case
- Describe the expected behavior
- Suggest an implementation approach if possible

### Code Contributions

Areas where contributions are welcome:
- Performance optimizations
- Additional descriptor types
- Alternative matching strategies
- More comprehensive examples
- Improved documentation
- Bug fixes

## Code Review Process

1. All submissions require review
2. Maintainers will provide feedback
3. Address reviewer comments
4. Once approved, changes will be merged

## Questions?

Feel free to open an issue for:
- Questions about the code
- Clarification on documentation
- Discussion of potential improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
