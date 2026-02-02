# Contributing to NTech LLM Tuner

Thank you for considering contributing to NTech LLM Tuner. This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, GPU model)
- Relevant log output

### Suggesting Features

Feature requests are welcome. Please open an issue with:
- A clear description of the feature
- Why this feature would be useful
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Test your changes thoroughly
5. Commit with clear, descriptive messages
6. Push to your fork
7. Open a pull request

### Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and modular
- Update documentation when adding features

### Testing

Before submitting a pull request:
- Test on both GPU and CPU if possible
- Verify the UI remains responsive
- Check that configuration save/load works
- Ensure training completes without errors
- Test with different model sizes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/ntech-llm-tuner.git
cd ntech-llm-tuner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python ollama_trainer_v2.py
```

## Areas for Contribution

Current areas where contributions would be particularly valuable:

- Additional model format support
- Improved error handling and recovery
- Performance optimizations
- UI/UX enhancements
- Documentation improvements
- Cross-platform testing
- Additional quantization methods
- Dataset preprocessing tools

## Questions

If you have questions about contributing, feel free to open an issue labeled "question" or reach out through GitHub discussions.
