from __future__ import annotations

import pytest

from agent_tco import (
    CostParams,
    GridSearch,
    WorkflowConfig,
    StepConfig,
    pareto_frontier,
)
from agent_tco.grid import GridResult


@pytest.fixture
def simple_results(cost_params, config_j3, config_j9) -> list[GridResult]:
    gs = GridSearch(cost_params=cost_params, n_runs=50, seed=42)
    results = gs.evaluate_all([config_j3, config_j9])
    pareto_frontier(results)
    return results


class TestGridResultToDict:
    def test_returns_dict(self, simple_results):
        d = simple_results[0].to_dict()
        assert isinstance(d, dict)

    def test_contains_core_keys(self, simple_results):
        d = simple_results[0].to_dict()
        for key in ("config_id", "n_runs", "acr", "expected_tco", "executive_metric", "pareto_dominated"):
            assert key in d, f"Missing key: {key}"

    def test_contains_scenario_keys(self, simple_results):
        d = simple_results[0].to_dict()
        for scenario_val in ("i1", "i2", "i3", "i4", "i5", "i6"):
            assert f"p_{scenario_val}" in d
            assert f"ci_lo_{scenario_val}" in d
            assert f"ci_hi_{scenario_val}" in d
            assert f"cost_{scenario_val}" in d

    def test_config_id_matches(self, simple_results):
        for r in simple_results:
            assert r.to_dict()["config_id"] == r.config.config_id

    def test_acr_value_matches(self, simple_results):
        for r in simple_results:
            assert r.to_dict()["acr"] == pytest.approx(r.acr)

    def test_pareto_dominated_flag_present(self, simple_results):
        d_list = [r.to_dict() for r in simple_results]
        dominated_count = sum(1 for d in d_list if d["pareto_dominated"])
        # At least one config should be dominated (j3 dominated by j9)
        assert dominated_count >= 1


class TestGridResultToDataframe:
    def test_returns_dataframe(self, simple_results):
        import pandas as pd
        df = GridResult.to_dataframe(simple_results)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches(self, simple_results):
        import pandas as pd
        df = GridResult.to_dataframe(simple_results)
        assert len(df) == len(simple_results)

    def test_column_count_nonzero(self, simple_results):
        import pandas as pd
        df = GridResult.to_dataframe(simple_results)
        assert len(df.columns) > 6  # at least core cols + scenario cols

    def test_config_ids_in_index(self, simple_results):
        import pandas as pd
        df = GridResult.to_dataframe(simple_results)
        assert set(df["config_id"]) == {r.config.config_id for r in simple_results}

    def test_acr_column_in_range(self, simple_results):
        import pandas as pd
        df = GridResult.to_dataframe(simple_results)
        assert (df["acr"] >= 0).all() and (df["acr"] <= 1).all()
