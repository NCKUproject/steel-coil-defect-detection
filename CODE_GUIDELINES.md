# Code Guidelines

## 1. No Magic Numbers / Strings
- Do not hardcode numbers/strings in code.
- Use named constants from `src/utils/constants.py` or configuration files in `configs/`.
- Each constant must have a meaning and documentation.

**Example (Good)**
```python
from src.utils.constants import RANDOM_SEED
set_global_seed(RANDOM_SEED)
```
**Example (Bad)**
```python
set_global_seed(42)  # ✗ magic number
```

## 2. Comments Placement
- Write comments above the code they describe.
- Do not use end-of-line comments (trailing comments), except for # noqa, # nosec, or URLs.

**Example (Good)**
```python
# Normalize images to ImageNet statistics.
x = normalize_imagenet(x)
```
**Example (Bad)**
```python
x = normalize_imagenet(x)  # ✗ trailing comment
```

## 3. Function Docstrings
- Every `function/method/class` must have a docstring with: `summary (one line)`,` Args`, `Returns`, `Raises (if any)`.
- Use Google style docstrings.

**Template**
```python
def example(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of a and b.
    """
    return a + b
```

## 4. Typing & Style
- Use Python 3.10+ type hints.
- Keep functions small and single-responsibility.
- Do not commit notebooks with outputs; clear outputs before commit.
- Use `logging` (not `print`) outside quick scripts.

## 5. Configuration First
- Hyperparameters and paths live in YAML under configs/.
- Code reads values via structured config loaders; no literals in code.

## 6. Tests/Sanity Checks
- Add quick sanity scripts under `src/**/sanity_*.py`.
