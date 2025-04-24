# Directory Structure Transition

## Important Notice

**As of April 24, 2025, the duplicate `fedzk/` directory at the project root has been deprecated.**

## Reason for Deprecation

The project initially had a non-standard structure with code in two locations:
- `fedzk/` at the project root
- `src/fedzk/` following standard Python packaging practices

This created confusion for developers and potential issues with imports, testing, and deployment.

## New Structure

The project now follows standard Python packaging practices:
- `src/fedzk/` contains all package code
- `tests/` contains all test files
- `docs/` contains documentation
- `examples/` contains usage examples

## Migration Plan

1. All new code should be added to `src/fedzk/` only
2. Existing code in `fedzk/` is being gradually migrated to `src/fedzk/`
3. The `fedzk/` directory will eventually be removed entirely
4. CI pipelines have been updated to use `src/fedzk/` for builds and tests

## Import Updates

If you were previously importing from the root fedzk directory:

```python
# Old import (DEPRECATED)
from fedzk.module import function

# New import (CORRECT)
from fedzk.module import function  # Python will resolve this to src/fedzk
```

The package has been configured so imports should work the same way, but now properly using the `src/fedzk` directory.

## Transition Assistance

If you encounter any issues during this transition, please:
1. Create an issue on GitHub
2. Mention the specific module or functionality affected
3. Include any error messages and steps to reproduce
