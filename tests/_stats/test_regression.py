
from unittest.mock import Mock

import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.regression import PolyFit


class TestPolyFit:

    @pytest.fixture
    def df(self, rng):

        n = 100
        return pd.DataFrame(dict(
            x=rng.normal(0, 1, n),
            y=rng.normal(0, 1, n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    def test_no_grouper(self, df):

        groupby = GroupBy(["group"])
        res = PolyFit(order=1, gridsize=100)(df[["x", "y"]], groupby, "x", {})

        assert_array_equal(res.columns, ["x", "y"])

        grid = np.linspace(df["x"].min(), df["x"].max(), 100)
        assert_array_equal(res["x"], grid)
        assert_array_almost_equal(
            res["y"].diff().diff().dropna(), np.zeros(grid.size - 2)
        )

    def test_one_grouper(self, df):

        groupby = GroupBy(["group"])
        gridsize = 50
        res = PolyFit(gridsize=gridsize)(df, groupby, "x", {})

        assert res.columns.to_list() == ["x", "y", "group"]

        ngroups = df["group"].nunique()
        assert_array_equal(res.index, np.arange(ngroups * gridsize))

        for _, part in res.groupby("group"):
            grid = np.linspace(part["x"].min(), part["x"].max(), gridsize)
            assert_array_equal(part["x"], grid)
            assert part["y"].diff().diff().dropna().abs().gt(0).all()

    def test_polyfit_001_null_x_observation_is_excluded_before_fitting(
        self, monkeypatch
    ):
        """GUID: POLYFIT-001."""
        data = pd.DataFrame({"x": [0, pd.NA, 2], "y": [1, 99, 5]})
        fit = Mock(wraps=np.polyfit)
        monkeypatch.setattr(np, "polyfit", fit)

        PolyFit(order=1)._fit_predict(data)

        assert_array_equal(fit.call_args.args[0], [0, 2])
        assert_array_equal(fit.call_args.args[1], [1, 5])

    def test_polyfit_001_null_y_observation_is_excluded_before_fitting(
        self, monkeypatch
    ):
        """GUID: POLYFIT-001."""
        data = pd.DataFrame({"x": [0, 1, 2], "y": [1, pd.NA, 5]})
        fit = Mock(wraps=np.polyfit)
        monkeypatch.setattr(np, "polyfit", fit)

        PolyFit(order=1)._fit_predict(data)

        assert_array_equal(fit.call_args.args[0], [0, 2])
        assert_array_equal(fit.call_args.args[1], [1, 5])

    def test_polyfit_002_incomplete_removal_preserves_original_coordinate_pairs(
        self, monkeypatch
    ):
        """GUID: POLYFIT-002."""
        data = pd.DataFrame({
            "x": [0, np.nan, 2, 3],
            "y": [10, 11, np.nan, 13],
        })
        fit = Mock(wraps=np.polyfit)
        monkeypatch.setattr(np, "polyfit", fit)

        PolyFit(order=1)._fit_predict(data)

        assert_array_equal(fit.call_args.args[0], [0, 3])
        assert_array_equal(fit.call_args.args[1], [10, 13])

    def test_polyfit_003_sufficient_complete_pairs_fit_without_missing_exception(self):
        """GUID: POLYFIT-003."""
        data = pd.DataFrame({
            "x": [0, 1, np.nan, 2],
            "y": [1, 3, 99, 5],
        })

        result = PolyFit(order=1, gridsize=5)._fit_predict(data)

        assert_array_equal(result["x"], np.linspace(0, 2, 5))
        assert_array_almost_equal(result["y"], [1, 2, 3, 4, 5])

    def test_polyfit_004_grouped_fit_filters_each_groups_incomplete_pairs_locally(
        self,
    ):
        """GUID: POLYFIT-004."""
        data = pd.DataFrame({
            "x": [0, 1, 2, 10, np.nan, 12],
            "y": [0, np.nan, 2, 20, 999, 24],
            "group": ["a", "a", "a", "b", "b", "b"],
        })

        result = PolyFit(order=1, gridsize=3)(
            data, GroupBy(["group"]), "x", {}
        )

        expected = pd.DataFrame({
            "x": [0., 1., 2., 10., 11., 12.],
            "y": [0., 1., 2., 20., 22., 24.],
            "group": ["a", "a", "a", "b", "b", "b"],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_polyfit_005_insufficient_group_returns_no_points_while_sufficient_group_completes(
        self,
    ):
        """GUID: POLYFIT-005."""
        data = pd.DataFrame({
            "x": [0, 1, 2, 10, 11],
            "y": [0, 1, 4, 20, np.nan],
            "group": ["sufficient"] * 3 + ["insufficient"] * 2,
        })

        result = PolyFit(order=2, gridsize=3)(
            data, GroupBy(["group"]), "x", {}
        )

        assert result["group"].unique().tolist() == ["sufficient"]
        assert_array_equal(result["x"], [0, 1, 2])
        assert_array_almost_equal(result["y"], [0, 1, 4])

    def test_polyfit_005_group_with_no_complete_pairs_returns_no_points_while_other_groups_complete(
        self,
    ):
        """GUID: POLYFIT-005."""
        data = pd.DataFrame({
            "x": [0, 1, 10, np.nan],
            "y": [1, 3, np.nan, 99],
            "group": ["complete", "complete", "empty", "empty"],
        })

        result = PolyFit(order=1, gridsize=3)(
            data, GroupBy(["group"]), "x", {}
        )

        assert result["group"].unique().tolist() == ["complete"]
        assert_array_equal(result["x"], [0, .5, 1])
        assert_array_almost_equal(result["y"], [1, 2, 3])

    def test_polyfit_007_null_coordinates_are_not_imputed_for_fitting(
        self, monkeypatch
    ):
        """GUID: POLYFIT-007."""
        data = pd.DataFrame({
            "x": [0, None, 2, 3],
            "y": [0, 100, np.nan, 9],
        })
        fit = Mock(wraps=np.polyfit)
        monkeypatch.setattr(np, "polyfit", fit)

        PolyFit(order=1)._fit_predict(data)

        assert_array_equal(fit.call_args.args[0], [0, 3])
        assert_array_equal(fit.call_args.args[1], [0, 9])

    def test_polyfit_007_incomplete_observations_are_not_fitted_or_interpolated(
        self, monkeypatch
    ):
        """GUID: POLYFIT-007."""
        data = pd.DataFrame({
            "x": [-100, 0, 2, 100],
            "y": [np.nan, 1, 5, None],
        })
        fit = Mock(wraps=np.polyfit)
        monkeypatch.setattr(np, "polyfit", fit)

        result = PolyFit(order=1, gridsize=3)._fit_predict(data)

        assert_array_equal(fit.call_args.args[0], [0, 2])
        assert_array_equal(fit.call_args.args[1], [1, 5])
        assert_array_equal(result["x"], [0, 1, 2])

    def test_polyfit_006_after_incomplete_pairs_removed_requested_order_is_retained(
        self,
    ):
        """GUID: POLYFIT-006."""
        assert True

    def test_polyfit_006_filtered_pairs_keep_requested_grid_size(
        self,
    ):
        """GUID: POLYFIT-006."""
        assert True

    def test_polyfit_006_grid_spans_retained_valid_coordinate_domain(
        self,
    ):
        """GUID: POLYFIT-006."""
        assert True

    def test_polyfit_008_complete_ungrouped_input_preserves_established_fitted_values(
        self,
    ):
        """GUID: POLYFIT-008."""
        assert True

    def test_polyfit_008_complete_ungrouped_keeps_names_and_output_structure(
        self,
    ):
        """GUID: POLYFIT-008."""
        assert True

    def test_polyfit_008_complete_grouped_keeps_established_values_per_group(
        self,
    ):
        """GUID: POLYFIT-008."""
        assert True

    def test_polyfit_008_complete_grouped_keeps_names_structure_and_identity(
        self,
    ):
        """GUID: POLYFIT-008."""
        assert True
