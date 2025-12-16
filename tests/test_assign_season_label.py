#!/usr/bin/env python

"""
Comprehensive test suite for assign_season_label() function.

Tests cover:
- Single month seasons (1-12)
- Multi-month seasons (2-5 months)
- Year wrap-around cases (December to January)
- Input validation (invalid months, wrong length)
- Error handling
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import pyseasonal
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyseasonal.utils.functions_seasonal import assign_season_label


class TestSingleMonthSeasons:
    """Test cases for single month seasons."""

    def test_january(self):
        assert assign_season_label([1]) == "JAN"

    def test_february(self):
        assert assign_season_label([2]) == "FEB"

    def test_march(self):
        assert assign_season_label([3]) == "MAR"

    def test_april(self):
        assert assign_season_label([4]) == "APR"

    def test_may(self):
        assert assign_season_label([5]) == "MAY"

    def test_june(self):
        assert assign_season_label([6]) == "JUN"

    def test_july(self):
        assert assign_season_label([7]) == "JUL"

    def test_august(self):
        assert assign_season_label([8]) == "AUG"

    def test_september(self):
        assert assign_season_label([9]) == "SEP"

    def test_october(self):
        assert assign_season_label([10]) == "OCT"

    def test_november(self):
        assert assign_season_label([11]) == "NOV"

    def test_december(self):
        assert assign_season_label([12]) == "DEC"


class TestTwoMonthSeasons:
    """Test cases for two consecutive month seasons."""

    def test_january_february(self):
        assert assign_season_label([1, 2]) == "JF"

    def test_february_march(self):
        assert assign_season_label([2, 3]) == "FM"

    def test_march_april(self):
        assert assign_season_label([3, 4]) == "MA"

    def test_april_may(self):
        assert assign_season_label([4, 5]) == "AM"

    def test_may_june(self):
        assert assign_season_label([5, 6]) == "MJ"

    def test_june_july(self):
        assert assign_season_label([6, 7]) == "JJ"

    def test_july_august(self):
        assert assign_season_label([7, 8]) == "JA"

    def test_august_september(self):
        assert assign_season_label([8, 9]) == "AS"

    def test_september_october(self):
        assert assign_season_label([9, 10]) == "SO"

    def test_october_november(self):
        assert assign_season_label([10, 11]) == "ON"

    def test_november_december(self):
        assert assign_season_label([11, 12]) == "ND"

    def test_december_january_wrap(self):
        """Critical: Test year wrap-around from December to January."""
        assert assign_season_label([12, 1]) == "DJ"


class TestThreeMonthSeasons:
    """Test cases for three consecutive month seasons."""

    def test_jfm(self):
        assert assign_season_label([1, 2, 3]) == "JFM"

    def test_fma(self):
        assert assign_season_label([2, 3, 4]) == "FMA"

    def test_mam(self):
        assert assign_season_label([3, 4, 5]) == "MAM"

    def test_amj(self):
        assert assign_season_label([4, 5, 6]) == "AMJ"

    def test_mjj(self):
        assert assign_season_label([5, 6, 7]) == "MJJ"

    def test_jja(self):
        assert assign_season_label([6, 7, 8]) == "JJA"

    def test_jas(self):
        assert assign_season_label([7, 8, 9]) == "JAS"

    def test_aso(self):
        assert assign_season_label([8, 9, 10]) == "ASO"

    def test_son(self):
        assert assign_season_label([9, 10, 11]) == "SON"

    def test_ond(self):
        assert assign_season_label([10, 11, 12]) == "OND"

    def test_ndj_wrap(self):
        """Critical: Test year wrap-around November-December-January."""
        assert assign_season_label([11, 12, 1]) == "NDJ"

    def test_djf_wrap(self):
        """Critical: Test year wrap-around December-January-February."""
        assert assign_season_label([12, 1, 2]) == "DJF"


class TestFourMonthSeasons:
    """Test cases for four consecutive month seasons."""

    def test_jfma(self):
        assert assign_season_label([1, 2, 3, 4]) == "JFMA"

    def test_fmam(self):
        assert assign_season_label([2, 3, 4, 5]) == "FMAM"

    def test_mamj(self):
        assert assign_season_label([3, 4, 5, 6]) == "MAMJ"

    def test_amjj(self):
        assert assign_season_label([4, 5, 6, 7]) == "AMJJ"

    def test_mjja(self):
        assert assign_season_label([5, 6, 7, 8]) == "MJJA"

    def test_jjas(self):
        assert assign_season_label([6, 7, 8, 9]) == "JJAS"

    def test_jaso(self):
        assert assign_season_label([7, 8, 9, 10]) == "JASO"

    def test_ason(self):
        assert assign_season_label([8, 9, 10, 11]) == "ASON"

    def test_sond(self):
        assert assign_season_label([9, 10, 11, 12]) == "SOND"

    def test_ondj_wrap(self):
        """Critical: Test year wrap-around October-November-December-January."""
        assert assign_season_label([10, 11, 12, 1]) == "ONDJ"

    def test_ndjf_wrap(self):
        """Critical: Test year wrap-around November-December-January-February."""
        assert assign_season_label([11, 12, 1, 2]) == "NDJF"

    def test_djfm_wrap(self):
        """Critical: Test year wrap-around December-January-February-March."""
        assert assign_season_label([12, 1, 2, 3]) == "DJFM"


class TestFiveMonthSeasons:
    """Test cases for five consecutive month seasons."""

    def test_jfmam(self):
        assert assign_season_label([1, 2, 3, 4, 5]) == "JFMAM"

    def test_fmamj(self):
        assert assign_season_label([2, 3, 4, 5, 6]) == "FMAMJ"

    def test_mamjj(self):
        assert assign_season_label([3, 4, 5, 6, 7]) == "MAMJJ"

    def test_amjja(self):
        assert assign_season_label([4, 5, 6, 7, 8]) == "AMJJA"

    def test_mjjas(self):
        assert assign_season_label([5, 6, 7, 8, 9]) == "MJJAS"

    def test_jjaso(self):
        assert assign_season_label([6, 7, 8, 9, 10]) == "JJASO"

    def test_jason(self):
        assert assign_season_label([7, 8, 9, 10, 11]) == "JASON"

    def test_asond(self):
        assert assign_season_label([8, 9, 10, 11, 12]) == "ASOND"

    def test_sondj_wrap(self):
        """Critical: Test year wrap-around September-October-November-December-January."""
        assert assign_season_label([9, 10, 11, 12, 1]) == "SONDJ"

    def test_ondjf_wrap(self):
        """Critical: Test year wrap-around October-November-December-January-February."""
        assert assign_season_label([10, 11, 12, 1, 2]) == "ONDJF"

    def test_ndjfm_wrap(self):
        """Critical: Test year wrap-around November-December-January-February-March."""
        assert assign_season_label([11, 12, 1, 2, 3]) == "NDJFM"

    def test_djfma_wrap(self):
        """Critical: Test year wrap-around December-January-February-March-April."""
        assert assign_season_label([12, 1, 2, 3, 4]) == "DJFMA"


class TestInvalidInputs:
    """Test cases for invalid inputs that should raise ValueError."""

    def test_empty_list(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="must contain between 1 and 5 months"):
            assign_season_label([])

    def test_too_many_months(self):
        """More than 5 months should raise ValueError."""
        with pytest.raises(ValueError, match="must contain between 1 and 5 months"):
            assign_season_label([1, 2, 3, 4, 5, 6])

    def test_invalid_month_zero(self):
        """Month 0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be integers between 1 and 12"):
            assign_season_label([0])

    def test_invalid_month_thirteen(self):
        """Month 13 should raise ValueError."""
        with pytest.raises(ValueError, match="must be integers between 1 and 12"):
            assign_season_label([13])

    def test_invalid_month_negative(self):
        """Negative month should raise ValueError."""
        with pytest.raises(ValueError, match="must be integers between 1 and 12"):
            assign_season_label([-1])

    def test_invalid_month_in_list(self):
        """Invalid month in middle of valid list should raise ValueError."""
        with pytest.raises(ValueError, match="must be integers between 1 and 12"):
            assign_season_label([1, 13, 3])

    def test_non_consecutive_months(self):
        """Non-consecutive months should raise ValueError."""
        with pytest.raises(ValueError, match="not consecutive"):
            assign_season_label([1, 3])

    def test_non_consecutive_months_gap(self):
        """Non-consecutive months with gap should raise ValueError."""
        with pytest.raises(ValueError, match="not consecutive"):
            assign_season_label([5, 6, 8])

    def test_wrong_wrap_around(self):
        """December not followed by January should raise ValueError."""
        with pytest.raises(ValueError, match="not consecutive"):
            assign_season_label([12, 2])

    def test_backwards_months(self):
        """Backwards sequence should raise ValueError."""
        with pytest.raises(ValueError, match="not consecutive"):
            assign_season_label([3, 2, 1])

    def test_duplicate_months(self):
        """Duplicate months should raise ValueError."""
        with pytest.raises(ValueError, match="not consecutive"):
            assign_season_label([1, 1])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_month_boundary_january(self):
        """First month of the year."""
        assert assign_season_label([1]) == "JAN"

    def test_single_month_boundary_december(self):
        """Last month of the year."""
        assert assign_season_label([12]) == "DEC"

    def test_full_year_end(self):
        """Last five months of the year."""
        assert assign_season_label([8, 9, 10, 11, 12]) == "ASOND"

    def test_year_start(self):
        """First five months of the year."""
        assert assign_season_label([1, 2, 3, 4, 5]) == "JFMAM"

    def test_mid_year(self):
        """Middle of the year."""
        assert assign_season_label([5, 6, 7, 8, 9]) == "MJJAS"


class TestReturnType:
    """Test that function returns correct type."""

    def test_returns_string(self):
        """Function should return a string."""
        result = assign_season_label([1])
        assert isinstance(result, str)

    def test_returns_uppercase(self):
        """Function should return uppercase labels."""
        result = assign_season_label([6, 7, 8])
        assert result.isupper()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
