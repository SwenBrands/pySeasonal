#!/usr/bin/env python

"""
Test suite for get_years_of_subperiod() function.

Tests cover:
- All valid subperiod keys (ENSO and QBO classifications)
- Return value type and structure validation
- Year list content validation
- Invalid input handling (KeyError for unknown keys)
- Console output verification (print messages)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from io import StringIO

# Add parent directory to path to import pyseasonal
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyseasonal.utils.functions_seasonal import get_years_of_subperiod


class TestValidSubperiods:
    """Test cases for all valid subperiod classifications."""

    def test_mod2strong_nino_oni(self):
        """Test moderate to strong El Niño years from ONI index."""
        years = get_years_of_subperiod('mod2strong_nino_oni')
        expected = [1982, 1983, 1986, 1987, 1991, 1992, 1997, 1998, 2009, 2010, 2015, 2016]
        assert years == expected
        assert len(years) == 12

    def test_mod2strong_nina_oni(self):
        """Test moderate to strong La Niña years from ONI index."""
        years = get_years_of_subperiod('mod2strong_nina_oni')
        expected = [1984, 1985, 1988, 1989, 1999, 2000, 2007, 2008, 2010, 2011, 2020, 2021, 2022]
        assert years == expected
        assert len(years) == 13

    def test_enso_nino_noaa(self):
        """Test El Niño years declared by NOAA."""
        years = get_years_of_subperiod('enso_nino_noaa')
        expected = [1983, 1987, 1988, 1992, 1995, 1998, 2003, 2007, 2010, 2016]
        assert years == expected
        assert len(years) == 10

    def test_enso_nina_noaa(self):
        """Test La Niña years declared by NOAA."""
        years = get_years_of_subperiod('enso_nina_noaa')
        expected = [1989, 1999, 2000, 2008, 2011, 2012, 2021, 2022]
        assert years == expected
        assert len(years) == 8

    def test_enso_neutral_noaa(self):
        """Test neutral ENSO years declared by NOAA."""
        years = get_years_of_subperiod('enso_neutral_noaa')
        expected = [
            1981, 1982, 1984, 1985, 1986, 1990, 1991, 1993, 1994, 1996, 1997, 2001,
            2002, 2004, 2005, 2006, 2009, 2013, 2014, 2015, 2017, 2018, 2019, 2020
        ]
        assert years == expected
        assert len(years) == 24

    def test_qbo50_pos(self):
        """Test positive QBO-50 years."""
        years = get_years_of_subperiod('qbo50_pos')
        expected = [
            1981, 1983, 1985, 1986, 1988, 1991, 1993, 1995, 1997, 1999,
            2000, 2002, 2004, 2009, 2011, 2014, 2017, 2019, 2021, 2023
        ]
        assert years == expected
        assert len(years) == 20

    def test_qbo50_neg(self):
        """Test negative QBO-50 years."""
        years = get_years_of_subperiod('qbo50_neg')
        expected = [
            1982, 1984, 1987, 1989, 1992, 1994, 1996, 1998, 2001,
            2003, 2005, 2007, 2010, 2012, 2015, 2018, 2022
        ]
        assert years == expected
        assert len(years) == 17

    def test_qbo50_trans(self):
        """Test transition QBO-50 years."""
        years = get_years_of_subperiod('qbo50_trans')
        expected = [1990, 2006, 2008, 2013, 2016, 2020]
        assert years == expected
        assert len(years) == 6

    def test_none_full_period(self):
        """Test 'none' returns full overlapping period as numpy array."""
        years = get_years_of_subperiod('none')
        expected = np.arange(1981, 2023, 1)
        assert isinstance(years, np.ndarray)
        np.testing.assert_array_equal(years, expected)
        assert len(years) == 42  # 1981 to 2022 inclusive


class TestReturnTypes:
    """Test return value types and structures."""

    def test_returns_list_for_enso(self):
        """Most subperiods return a list."""
        years = get_years_of_subperiod('mod2strong_nino_oni')
        assert isinstance(years, list)

    def test_returns_numpy_array_for_none(self):
        """The 'none' subperiod returns numpy array."""
        years = get_years_of_subperiod('none')
        assert isinstance(years, np.ndarray)

    def test_all_years_are_integers(self):
        """All year values should be integers."""
        for subperiod in ['mod2strong_nino_oni', 'enso_nina_noaa', 'qbo50_pos']:
            years = get_years_of_subperiod(subperiod)
            for year in years:
                assert isinstance(year, (int, np.integer))

    def test_years_in_valid_range(self):
        """All years should be between 1980 and 2025."""
        subperiods = [
            'mod2strong_nino_oni', 'mod2strong_nina_oni', 'enso_nino_noaa',
            'enso_nina_noaa', 'enso_neutral_noaa', 'qbo50_pos', 'qbo50_neg',
            'qbo50_trans', 'none'
        ]
        for subperiod in subperiods:
            years = get_years_of_subperiod(subperiod)
            for year in years:
                assert 1980 <= year <= 2025, f"Year {year} out of range for {subperiod}"


class TestYearProperties:
    """Test properties of returned year lists."""

    def test_years_are_sorted(self):
        """Year lists should be in ascending order."""
        subperiods = [
            'mod2strong_nino_oni', 'mod2strong_nina_oni', 'enso_nino_noaa',
            'enso_nina_noaa', 'qbo50_pos', 'qbo50_neg'
        ]
        for subperiod in subperiods:
            years = get_years_of_subperiod(subperiod)
            years_list = list(years)  # Convert numpy array to list if needed
            assert years_list == sorted(years_list), f"Years not sorted for {subperiod}"

    def test_no_duplicate_years(self):
        """Each year should appear only once in a list."""
        subperiods = [
            'mod2strong_nino_oni', 'mod2strong_nina_oni', 'enso_nino_noaa',
            'enso_nina_noaa', 'qbo50_pos', 'qbo50_neg'
        ]
        for subperiod in subperiods:
            years = get_years_of_subperiod(subperiod)
            years_list = list(years)
            assert len(years_list) == len(set(years_list)), f"Duplicates in {subperiod}"

    def test_enso_categories_mutually_exclusive(self):
        """ENSO categories should not overlap significantly."""
        nino = set(get_years_of_subperiod('enso_nino_noaa'))
        nina = set(get_years_of_subperiod('enso_nina_noaa'))
        neutral = set(get_years_of_subperiod('enso_neutral_noaa'))

        # No year should be in both El Niño and La Niña
        assert len(nino & nina) == 0, "El Niño and La Niña years overlap"

        # No year should be in both El Niño and neutral
        assert len(nino & neutral) == 0, "El Niño and neutral years overlap"

        # No year should be in both La Niña and neutral
        assert len(nina & neutral) == 0, "La Niña and neutral years overlap"

    def test_qbo_categories_cover_all_years(self):
        """QBO categories should cover most years (pos + neg + trans)."""
        pos = set(get_years_of_subperiod('qbo50_pos'))
        neg = set(get_years_of_subperiod('qbo50_neg'))
        trans = set(get_years_of_subperiod('qbo50_trans'))

        # Combined should cover all years from 1981-2023
        all_qbo = pos | neg | trans
        assert len(all_qbo) == 43, f"QBO categories don't cover expected years, got {len(all_qbo)}"


class TestInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_invalid_subperiod_raises_keyerror(self):
        """Unknown subperiod should raise KeyError."""
        with pytest.raises(KeyError, match="unknown entry"):
            get_years_of_subperiod('invalid_key')

    def test_empty_string_raises_keyerror(self):
        """Empty string should raise KeyError."""
        with pytest.raises(KeyError, match="unknown entry"):
            get_years_of_subperiod('')

    def test_typo_in_subperiod_raises_keyerror(self):
        """Typo in subperiod name should raise KeyError."""
        with pytest.raises(KeyError, match="unknown entry"):
            get_years_of_subperiod('mod2strong_nino_oni_typo')

    def test_case_sensitive(self):
        """Function should be case-sensitive."""
        with pytest.raises(KeyError):
            get_years_of_subperiod('MOD2STRONG_NINO_ONI')

    def test_none_uppercase_raises_keyerror(self):
        """'None' (capitalized) should raise KeyError."""
        with pytest.raises(KeyError):
            get_years_of_subperiod('None')


class TestPrintOutput:
    """Test that function prints expected messages."""

    def test_prints_message_for_nino(self, capsys):
        """Function should print verification message for El Niño."""
        get_years_of_subperiod('mod2strong_nino_oni')
        captured = capsys.readouterr()
        assert 'El Niño' in captured.out
        assert 'ONI index' in captured.out

    def test_prints_message_for_nina(self, capsys):
        """Function should print verification message for La Niña."""
        get_years_of_subperiod('mod2strong_nina_oni')
        captured = capsys.readouterr()
        assert 'La Niña' in captured.out
        assert 'ONI index' in captured.out

    def test_prints_years_in_message(self, capsys):
        """Function should include year list in printed message."""
        years = get_years_of_subperiod('qbo50_trans')
        captured = capsys.readouterr()
        # Check that some years from the list appear in output
        assert '1990' in captured.out or '2006' in captured.out

    def test_prints_message_for_none(self, capsys):
        """Function should print message about full period for 'none'."""
        get_years_of_subperiod('none')
        captured = capsys.readouterr()
        assert 'full overlapping period' in captured.out.lower()


class TestDataConsistency:
    """Test consistency of climate classification data."""

    def test_mod2strong_includes_strong_events(self):
        """Moderate to strong should include well-known strong events."""
        nino_years = get_years_of_subperiod('mod2strong_nino_oni')
        # 1997-1998 and 2015-2016 were very strong El Niño events
        assert 1997 in nino_years
        assert 1998 in nino_years
        assert 2015 in nino_years
        assert 2016 in nino_years

    def test_nina_includes_recent_events(self):
        """La Niña list should include recent multi-year event."""
        nina_years = get_years_of_subperiod('mod2strong_nina_oni')
        # 2020-2022 was a prolonged La Niña event
        assert 2020 in nina_years
        assert 2021 in nina_years
        assert 2022 in nina_years

    def test_none_range_matches_documentation(self):
        """'none' should return 1981-2022 as documented."""
        years = get_years_of_subperiod('none')
        assert years[0] == 1981
        assert years[-1] == 2022
        assert len(years) == 42


class TestAllValidKeys:
    """Verify all documented subperiod keys work."""

    @pytest.mark.parametrize("subperiod", [
        'mod2strong_nino_oni',
        'mod2strong_nina_oni',
        'enso_nino_noaa',
        'enso_nina_noaa',
        'enso_neutral_noaa',
        'qbo50_pos',
        'qbo50_neg',
        'qbo50_trans',
        'none'
    ])
    def test_all_valid_keys_work(self, subperiod):
        """All documented subperiod keys should work without error."""
        years = get_years_of_subperiod(subperiod)
        assert years is not None
        assert len(years) > 0


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
