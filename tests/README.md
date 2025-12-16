# pySeasonal Tests

This directory contains the test suite for the pySeasonal package.

## Test Files

### `test_assign_season_label.py`

Comprehensive test suite for the `assign_season_label()` function with **78 test cases** covering:

### `test_get_years_of_subperiod.py`

Test suite for the `get_years_of_subperiod()` function with **38 test cases** covering:

#### Test Coverage

1. **Valid Subperiods (9 tests)**
   - Tests all 9 valid subperiod classifications
   - Verifies correct year lists for ENSO (El Niño, La Niña, neutral)
   - Verifies correct year lists for QBO (positive, negative, transition)
   - Tests special 'none' case returning full period (1981-2022)

2. **Return Types (4 tests)**
   - Verifies lists returned for ENSO/QBO classifications
   - Verifies numpy array returned for 'none'
   - Ensures all years are integers
   - Validates years are in reasonable range (1980-2025)

3. **Year Properties (4 tests)**
   - Tests years are sorted in ascending order
   - Ensures no duplicate years in lists
   - Validates ENSO categories are mutually exclusive
   - Verifies QBO categories cover expected time period

4. **Invalid Inputs (5 tests)**
   - Tests KeyError raised for unknown subperiod keys
   - Tests case sensitivity of keys
   - Tests error handling for typos and empty strings

5. **Print Output (4 tests)**
   - Verifies correct console messages printed
   - Checks year lists appear in output
   - Validates messages mention correct climate phenomena

6. **Data Consistency (3 tests)**
   - Validates well-known climate events are included
   - Tests for major El Niño events (1997-98, 2015-16)
   - Tests for recent La Niña events (2020-2022)

7. **All Valid Keys (9 parametrized tests)**
   - Systematically tests each valid key works correctly
   - Uses pytest parametrize for comprehensive coverage

### `test_assign_season_label.py` (detailed)

#### Test Coverage

1. **Single Month Seasons (12 tests)**
   - Tests all 12 months individually
   - Verifies correct full month name output (e.g., 'JAN', 'FEB', 'DEC')

2. **Two Month Seasons (12 tests)**
   - Tests all consecutive 2-month combinations
   - **Includes critical year wrap-around test:** `[12, 1]` → `'DJ'`

3. **Three Month Seasons (12 tests)**
   - Tests all consecutive 3-month combinations
   - **Includes critical year wrap-around tests:**
     - `[11, 12, 1]` → `'NDJ'`
     - `[12, 1, 2]` → `'DJF'`

4. **Four Month Seasons (12 tests)**
   - Tests all consecutive 4-month combinations
   - **Includes critical year wrap-around tests:**
     - `[10, 11, 12, 1]` → `'ONDJ'`
     - `[11, 12, 1, 2]` → `'NDJF'`
     - `[12, 1, 2, 3]` → `'DJFM'`

5. **Five Month Seasons (12 tests)**
   - Tests all consecutive 5-month combinations
   - **Includes critical year wrap-around tests:**
     - `[9, 10, 11, 12, 1]` → `'SONDJ'`
     - `[10, 11, 12, 1, 2]` → `'ONDJF'`
     - `[11, 12, 1, 2, 3]` → `'NDJFM'`
     - `[12, 1, 2, 3, 4]` → `'DJFMA'`

6. **Invalid Inputs (11 tests)**
   - Empty list
   - Too many months (>5)
   - Invalid month values (0, 13, -1, etc.)
   - Non-consecutive months
   - Wrong wrap-around sequences
   - Backwards sequences
   - Duplicate months

7. **Edge Cases (5 tests)**
   - Boundary conditions (first/last months of year)
   - Full year coverage tests

8. **Return Type Validation (2 tests)**
   - Verifies return type is string
   - Verifies output is uppercase

## Installation

First, install the development dependencies:

```bash
# Using pip
pip install -e ".[dev]"

# Or using uv (if available)
uv pip install -e ".[dev]"
```

This will install pytest and pytest-cov along with the main package dependencies.

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_assign_season_label.py -v
```

### Run specific test class:
```bash
pytest tests/test_assign_season_label.py::TestSingleMonthSeasons -v
```

### Run specific test:
```bash
pytest tests/test_assign_season_label.py::TestTwoMonthSeasons::test_december_january_wrap -v
```

### Run with coverage:
```bash
pytest tests/ --cov=pyseasonal --cov-report=html
```

## Test Organization

Tests are organized into logical test classes:
- `TestSingleMonthSeasons` - Single month functionality
- `TestTwoMonthSeasons` - Two consecutive months
- `TestThreeMonthSeasons` - Three consecutive months
- `TestFourMonthSeasons` - Four consecutive months
- `TestFiveMonthSeasons` - Five consecutive months
- `TestInvalidInputs` - Error handling and validation
- `TestEdgeCases` - Boundary conditions
- `TestReturnType` - Output type validation

## Critical Tests

The most critical tests verify year wrap-around functionality (December → January transitions):
- ✅ `test_december_january_wrap` - Tests `[12, 1]` → `'DJ'`
- ✅ `test_ndj_wrap` - Tests `[11, 12, 1]` → `'NDJ'`
- ✅ `test_djf_wrap` - Tests `[12, 1, 2]` → `'DJF'`
- ✅ All 10 year wrap-around scenarios (2-5 month seasons)

These tests were added to prevent regression of the wrap-around logic bug that was fixed during refactoring.

## Test Results

Latest test run:
- **78 tests total**
- **78 passed** ✅
- **0 failed**
- **Test coverage:** 100% of `assign_season_label()` functionality

## Adding New Tests

When adding new tests:
1. Add test methods to appropriate test class
2. Follow naming convention: `test_<description>`
3. Use descriptive docstrings for critical tests
4. Verify tests pass: `pytest tests/test_assign_season_label.py -v`

## Dependencies

- pytest >= 8.0
- Python >= 3.10

Pytest is already included in the project dependencies.
