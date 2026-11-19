import json

import numpy as np
import pytest
from helpers import circle, map_file

import benchmark
import export_dataset
import main as main_script
import visualize
from diffdrive.evaluation import Suite, aggregate, format_table, read_results, run_suite


def _suite(short_line):
    return Suite.model_validate(
        {
            "name": "t",
            "seeds": [0, 1],
            "maps": [short_line],
            "scenarios": [{"scenario": "nominal", "sweep": {"sensor.latency": [0.0, 0.05]}}],
            "controllers": ["pid", {"controller": "pure_pursuit", "label": "pp"}, "lqr"],
        }
    )


def test_suite_expansion_is_paired(short_line):
    specs = _suite(short_line).expand()
    assert len(specs) == 2 * 3 * 2
    labels = {s.scenario_label for s in specs}
    assert labels == {
        f"{short_line}/nominal[sensor.latency=0.0]",
        f"{short_line}/nominal[sensor.latency=0.05]",
    }
    assert all(s.scenario.map == short_line for s in specs)


def test_suite_maps_axis_and_per_scenario_override():
    s = Suite.model_validate(
        {
            "name": "m",
            "maps": ["map00", "map01"],
            "scenarios": ["nominal", {"scenario": "lateral_offset", "maps": ["map02"]}, "noisy"],
            "controllers": ["pid"],
        }
    )
    labels = [r.scenario_label for r in s.expand()]
    assert labels == [
        "map00/nominal",
        "map01/nominal",
        "map02/lateral_offset",
        "map00/noisy",
        "map01/noisy",
    ]
    # without any maps a scenario uses its own default map
    d = Suite.model_validate({"name": "d", "scenarios": ["heading_offset"], "controllers": ["pid"]})
    assert [r.scenario_label for r in d.expand()] == ["map00/heading_offset"]


def test_run_suite_writes_results(tmp_path, short_line):
    rows = run_suite(_suite(short_line), tmp_path, jobs=1)
    assert (tmp_path / "results.csv").is_file()
    back = read_results(tmp_path)
    assert len(back) == len(rows) == 12
    assert all(r["map"] == short_line for r in back)
    lqr = [r for r in back if r["controller"] == "lqr"]
    assert all(r["termination"] == "controller_error" for r in lqr)  # templates don't crash
    table = format_table(aggregate(back))
    assert "pp" in table and "pid" in table


def test_duplicate_controller_labels_rejected():
    s = Suite.model_validate({"name": "d", "scenarios": ["nominal"], "controllers": ["pid", "pid"]})
    with pytest.raises(ValueError, match="duplicate controller"):
        s.expand()


def test_builtin_suites_expand():
    for name in ("smoke", "nominal", "latency", "noise", "mismatch", "dataset"):
        assert Suite.from_yaml(name).expand()


# ------------------------------------------------------------------- scripts
def _run(module, argv):
    return module.main(module.build_parser().parse_args(argv))


def test_main_list_and_print_only(tmp_path, capsys, short_line):
    _run(main_script, ["--list"])
    out = capsys.readouterr().out
    assert "pure_pursuit" in out and "map02" in out and "noisy" in out
    assert _run(main_script, ["--map", short_line, "--out", str(tmp_path)]) is None
    assert "pos RMSE" in capsys.readouterr().out
    assert not any(tmp_path.iterdir())  # nothing saved without --save


def test_main_save_and_visualize(tmp_path, short_line):
    a = _run(main_script, ["--map", short_line, "--save", "--out", str(tmp_path)])
    assert {p.name for p in a.iterdir()} == {
        "episode.npz",
        "manifest.json",
        "summary.png",
        "animation.gif",
    }
    b = _run(
        main_script,
        ["--map", short_line, "--controller", "pure_pursuit", "--save", "--no-gif",
         "--out", str(tmp_path)],
    )  # fmt: skip
    assert not (b / "animation.gif").exists()
    _run(visualize, [str(a), str(b), "--output", str(tmp_path / "cmp.png")])
    assert (tmp_path / "cmp.png").is_file()


def test_benchmark_report_and_dataset_export(tmp_path, maps_dir):
    m = map_file(maps_dir, "quarter_circle", circle(1.0, laps=0.25))
    suite = tmp_path / "s.yaml"
    suite.write_text(
        f"name: s\nseeds: [0, 1, 2]\nmaps: ['{m}']\n"
        "scenarios: [nominal, noisy]\ncontrollers: [pure_pursuit, pid]\n"
    )
    out = tmp_path / "bench"
    _run(benchmark, ["--suite", str(suite), "--out", str(out), "--report"])
    assert (out / "results.csv").is_file()
    assert (out / "figures" / "bars_pos_rmse.png").is_file()
    _run(visualize, [str(out), "--metric", "cte_rmse"])
    assert (out / "figures" / "bars_cte_rmse.png").is_file()

    ds = tmp_path / "ds"
    _run(export_dataset, [str(out), "--out", str(ds), "--controller", "pure_pursuit"])
    d = np.load(ds / "data.npz")
    meta = json.loads((ds / "meta.json").read_text())
    assert d["X"].shape == (len(d["Y"]), len(export_dataset.FEATURES))
    assert np.isfinite(d["X"]).all() and np.isfinite(d["Y"]).all()
    assert len(meta["episodes"]) == 6 and {e["controller"] for e in meta["episodes"]} == {
        "pure_pursuit"
    }
    # splits are per episode
    for e in np.unique(d["episode"]):
        assert len(np.unique(d["split"][d["episode"] == e])) == 1
