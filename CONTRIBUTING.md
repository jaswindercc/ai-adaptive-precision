# Contributing to Adaptive Precision Scaling

Thanks for your interest in contributing! This project implements the Adaptive Precision Scaling System from the paper *"Low-Cost AI Infrastructure: Strategies and Optimization"* by [Jaswinder Singh](https://jaswinder.cc/).

## Getting Started

1. **Fork & clone** the repository
2. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

1. Create a feature branch from `main`
2. Write your code with tests
3. Run lint and tests:
   ```bash
   ruff check src/ tests/
   pytest tests/ -v --cov=adaptive_precision
   ```
4. Submit a pull request

## What to Contribute

- **New precision strategies** — Add to `controller.py` via the `Strategy` enum
- **Framework wrappers** — Add PyTorch, TensorFlow, or ONNX wrapper improvements
- **Hardware monitors** — Extend monitoring for AMD GPUs, TPUs, or edge devices
- **Benchmarks** — Real-world benchmarks with actual models
- **Documentation** — Examples, tutorials, blog posts

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting
- Follow existing patterns in the codebase
- Keep functions focused and well-named
- Add tests for new functionality

## Reporting Issues

Use [GitHub Issues](https://github.com/jaswindercc/ai-adaptive-precision/issues) with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Python version, GPU)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
