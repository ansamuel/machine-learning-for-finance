# Machine Learning for Finance

Fork of the Momentum Transformer implementation ([repo](https://github.com/kieranjwood/trading-momentum-transformer.git)).

## Installation

This project uses uv for dependency management. To install:

```bash
uv venv # Create virtual environment
uv pip install -e . # Install dependencies
source .venv/bin/activate # Activate virtual environment
```

## Project Structure and Guidelines

### Module Organization
- All code is organized into domain-specific top-level modules
- Each module has at most one nested level for ease of access
- Modules are designed to be as independent as possible

#### Directory Structure
```
data/               # Raw data loading and initial processing
preprocessing/      # Time series windowing, scaling, and ML preparation
features/           # Financial feature engineering and transformations
models/             # Model architectures, custom losses, and training
strategies/         # Trading strategy implementations
backtesting/        # Framework for strategy evaluation
utils/              # Shared utilities across the codebase
scripts/            # Experiment runners and entry points
```

#### Refactoring Progress
- [x] `data/` - Initial structure implemented
- [x] `features/` - Initial structure implemented
- [x] `utils/` - Initial structure implemented
- [x] `scripts/` - Initial structure implemented
- [x] `strategies/` - Created momentum module, need experimental module
- [ ] `preprocessing/` - Need to implement from ModelFeatures
- [ ] `models/` - Need to implement from mlmomentum
- [ ] `backtesting/` - Need to implement from backtest.py

#### Next Steps
1. Create `preprocessing/` module:
   - Extract windowing logic from ModelFeatures._batch_data
   - Implement time series scaling appropriate for financial data
   - Create train/validation/test splitting with time boundaries

2. Implement `models/` module:
   - Move LSTM and Transformer architectures from mlmomentum
   - Extract SharpeLoss implementation to models/losses.py
   - Create standardized training loops in models/training.py

3. Develop `backtesting/` module:
   - Extract performance evaluation from backtest.py
   - Implement rolling window evaluation framework
   - Create metrics specific to trading strategies

4. Create `strategies/experimental/`:
   - Implement deep momentum network strategy
   - Add transformer-based trading strategy 

5. Update training scripts to use new module structure

### Code Style
- Prioritize readability over excessive abstraction
- Avoid nested functions; use sequential code or lambdas when appropriate
- Use modern Python type hints with concise docstrings
- Function documentation should focus on "what" and "why" rather than duplicating type information
- Format code with `ruff format` and lint with `ruff check`
- Follow DRY principles without sacrificing readability
- Line length: 88 characters

### Naming Conventions
- Functions: Use `<verb>_<object>` pattern in snake_case (e.g., `calculate_returns`, `load_prices`)
- Variables: Descriptive snake_case names (e.g., `prices`, `daily_returns`)
- Constants: ALL_CAPS with underscores at module level (e.g., `WINSORIZE_THRESHOLD`, `LOOKBACK_DAYS`)
- Classes: PascalCase names (e.g., `MomentumStrategy`, `DataProcessor`)
- Type variables: PascalCase, can be single letters (e.g., `T`, `DataFrameType`)
- Collections: Use plural forms (e.g., `returns` rather than `return_list`)
- DataFrames: Use plural nouns without suffixes (e.g., `positions`, `returns`, not `positions_df`)
- Series: Use singular nouns or distinguishing names (e.g., `price_series`, `return_values`)
- Boolean variables: Prefix with `is_`, `has_`, etc. (e.g., `is_valid`, `has_data`)
- Avoid single-letter variables except for indices or mathematical notation
- Avoid redundant suffixes when type information is provided by type hints
- Avoid generic variable names like `data` or `*_data` - use specific terms that describe what the data represents (e.g., `prices`, `volatilities`, `momentum_signals`)
- Use finance domain terminology consistently (e.g., `vol` for volatility is acceptable)

### Development Focus
- Migrating to latest TensorFlow/Keras
- Maintaining clear code structure during development
- Deferring excessive modularity until later stages