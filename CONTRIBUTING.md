# Contributing to Stock Prediction LSTM

We welcome contributions to the Stock Prediction LSTM project! By contributing, you help us improve and expand the capabilities of this stock prediction system.

## How to Contribute

There are several ways you can contribute:

1.  **Report Bugs**: If you find a bug, please open an issue on our [GitHub Issues page](https://github.com/yourusername/stock_prediction_lstm/issues). Provide a clear and concise description of the bug, steps to reproduce it, and expected behavior.

2.  **Suggest Features**: Have an idea for a new feature or improvement? Open an issue on GitHub to discuss it. We appreciate all suggestions that can make the project better.

3.  **Submit Pull Requests**: If you've fixed a bug, implemented a new feature, or made an improvement, we encourage you to submit a pull request. Please follow the guidelines below.

4.  **Improve Documentation**: Good documentation is crucial. If you find areas that can be improved, whether it's in the README, code comments, or other documentation, feel free to submit a pull request.

5.  **Provide Feedback**: Share your thoughts and experiences with the project. Your feedback helps us understand how the tool is being used and where it can be enhanced.

## Pull Request Guidelines

To ensure a smooth and efficient review process, please follow these guidelines when submitting pull requests:

1.  **Fork the Repository**: Start by forking the `stock_prediction_lstm` repository to your GitHub account.

2.  **Create a New Branch**: Create a new branch for your changes. Use a descriptive name that reflects the nature of your work (e.g., `feature/new-sentiment-source`, `bugfix/api-key-issue`).

    ```bash
    git checkout -b feature/your-feature-name
    ```

3.  **Code Style**: Adhere to the existing code style and conventions of the project. We use `black` for code formatting and `flake8` for linting. Please ensure your code passes these checks.

    ```bash
    # Install development dependencies (if you haven't already)
    pip install -e ".[dev]"

    # Run black to format your code
    black .

    # Run flake8 to check for linting errors
    flake8 .
    ```

4.  **Add Tests**: If you're adding new features or fixing bugs, please include appropriate unit tests to cover your changes. This helps ensure the stability and correctness of the codebase.

    ```bash
    pytest
    ```

5.  **Update Documentation**: If your changes introduce new features, modify existing behavior, or fix a bug that impacts users, please update the relevant documentation (e.g., `README.md`, docstrings).

6.  **Commit Messages**: Write clear, concise, and descriptive commit messages. A good commit message explains *what* was changed and *why*.

7.  **Open a Pull Request**: Once your changes are ready, open a pull request to the `main` branch of the `stock_prediction_lstm` repository. Provide a detailed description of your changes, including:
    *   A summary of the changes.
    *   The motivation behind the changes.
    *   Any relevant issue numbers (e.g., `Fixes #123`, `Closes #456`).
    *   How to test your changes (if applicable).

## Development Setup

To set up your development environment, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/stock_prediction_lstm.git
    cd stock_prediction_lstm
    ```

2.  **Install dependencies** (including development dependencies):

    ```bash
    pip install -e ".[dev]"
    ```

3.  **Install pre-commit hooks** (optional, but recommended):

    ```bash
    pre-commit install
    ```
    This will automatically run `black` and `flake8` checks before each commit.

## Code of Conduct

We adhere to a [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming and inclusive environment for all contributors. Please review it before contributing.

## License

By contributing to Stock Prediction LSTM, you agree that your contributions will be licensed under the [MIT License](LICENSE).

Thank you for your interest in contributing to Stock Prediction LSTM!