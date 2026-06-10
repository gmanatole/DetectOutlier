import json
import numpy as np

from outlierdetect import Heuristic, ProfileInput, build_level_features, linear_detrend
from outlierdetect.training.synthetic import degrade_highres_profile


def test_linear_detrend_recovers_known_line():
    p = np.linspace(0, 100, 10)
    x = (p - p.min()) / (p.max() - p.min())
    residual = 0.2 - 0.05 * x
    fit = linear_detrend(p, residual, sigma=np.ones_like(p))
    assert np.isclose(fit.intercept, 0.2)
    assert np.isclose(fit.slope, -0.05)
    assert np.nanmax(np.abs(fit.residual)) < 1e-10


def test_profile_features_have_expected_shape():
    p = np.array([0, 10, 30, 60, 100], dtype=float)
    profile = ProfileInput(
        pressure=p,
        temperature=np.linspace(5, 2, p.size),
        salinity=np.linspace(34.2, 34.7, p.size),
        residual_t=np.zeros(p.size),
        residual_s=np.zeros(p.size),
        sigma_t=np.full(p.size, 0.2),
        sigma_s=np.full(p.size, 0.03),
        sigma_vert=np.full(p.size, 30.0),
    )
    feats = build_level_features(profile)
    assert feats.level_features.shape[0] == p.size
    assert "sigma_vert" in feats.feature_names
    assert "detrended_z_s" in feats.feature_names


def test_heuristic_predict_returns_object():
    p_hr = np.linspace(0, 500, 120)
    t_hr = 5 - 4 * (1 - np.exp(-p_hr / 120))
    s_hr = 34.1 + 0.6 * (1 - np.exp(-p_hr / 180))
    synth = degrade_highres_profile(p_hr, t_hr, s_hr, rng=np.random.default_rng(1))
    result = Heuristic().predict(synth.example.profile)
    assert 0 <= result.profile_bad_probability <= 1
    assert result.point_bad_t.shape == synth.example.profile.pressure.shape
    assert result.pressure_grid is not None
    assert result.temperature_reconstructed is not None
    assert result.salinity_reconstructed is not None


def test_predict_cli_rejects_plot_count():
    from outlierdetect.cli import _build_predict_parser

    parser = _build_predict_parser()

    try:
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--argo-root",
                "data",
                "--plot-count",
                "3",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected --plot-count to be rejected by predict")


def test_degrade_highres_profile_drops_negative_pressures():
    p_hr = np.array([-5.0, 0.0, 10.0, 20.0, 40.0, 80.0], dtype=float)
    t_hr = np.linspace(6.0, 2.0, p_hr.size)
    s_hr = np.linspace(34.0, 34.6, p_hr.size)

    synth = degrade_highres_profile(p_hr, t_hr, s_hr, rng=np.random.default_rng(2), grid_size=6)

    assert np.all(synth.example.profile.pressure >= 0.0)
    assert np.all(synth.truth_pressure_grid >= 0.0)
    assert synth.example.labels.pressure_grid is not None
    assert np.all(synth.example.labels.pressure_grid >= 0.0)


def test_sigma0_from_ts_uses_gsw(monkeypatch):
    import sys
    import types

    from outlierdetect.density import sigma0_from_ts

    fake_gsw = types.SimpleNamespace()

    def sa_from_sp(sp, pressure, lon, lat):
        return np.asarray(sp, dtype=float) + 0.5

    def ct_from_t(sa, temp, pressure):
        return np.asarray(temp, dtype=float) - 0.1

    def sigma0(sa, ct):
        return np.asarray(sa, dtype=float) - np.asarray(ct, dtype=float)

    fake_gsw.SA_from_SP = sa_from_sp
    fake_gsw.CT_from_t = ct_from_t
    fake_gsw.sigma0 = sigma0
    monkeypatch.setitem(sys.modules, "gsw", fake_gsw)

    out = sigma0_from_ts(np.array([34.0, 35.0]), np.array([10.0, 11.0]))
    assert np.allclose(out, np.array([24.6, 24.6]))


def test_argo_subsampling_stays_profile_specific():
    from outlierdetect.argo import ArgoProfile
    from outlierdetect.training.argo import build_argo_synthetic_examples

    profile_a = ArgoProfile(
        "a",
        np.array([0, 10, 30, 60, 120, 200, 320], dtype=float),
        np.linspace(6, 2, 7),
        np.linspace(34.0, 34.7, 7),
    )
    profile_b = ArgoProfile(
        "b",
        np.array([5, 15, 45, 90, 150, 240, 360, 500, 650], dtype=float),
        np.linspace(7, 1, 9),
        np.linspace(33.8, 35.0, 9),
    )

    synth = build_argo_synthetic_examples([profile_a, profile_b], n_examples_per_profile=1, n_levels=5, seed=2)
    pressures = [item.example.profile.pressure for item in synth]

    assert len(pressures) == 2
    assert pressures[0].shape == (5,)
    assert pressures[1].shape == (5,)
    assert not np.array_equal(pressures[0], pressures[1])
    assert np.all(np.isin(pressures[0], profile_a.pressure))
    assert np.all(np.isin(pressures[1], profile_b.pressure))


def test_argo_parquet_export_walks_recursively(tmp_path, monkeypatch):
    from pathlib import Path

    import pandas as pd

    from outlierdetect.argo import ArgoProfile
    from outlierdetect.parquet import argo_directory_to_dataframe, collect_nc_paths, write_argo_parquet

    root = tmp_path / "argo"
    nested = root / "sub" / "dir"
    nested.mkdir(parents=True)
    (root / "a.nc").write_text("", encoding="utf-8")
    (nested / "b.nc").write_text("", encoding="utf-8")
    (nested / "ignore.txt").write_text("", encoding="utf-8")

    paths = collect_nc_paths(root)
    assert [path.name for path in paths] == ["a.nc", "b.nc"]

    profiles_by_file = {
        str(root / "a.nc"): [
            ArgoProfile("a_001", np.array([0, 10], dtype=float), np.array([5.0, 4.5]), np.array([34.0, 34.1]))
        ],
        str(nested / "b.nc"): [
            ArgoProfile("b_001", np.array([0, 20, 40], dtype=float), np.array([6.0, 5.0, 4.0]), np.array([33.9, 34.0, 34.2])),
            ArgoProfile("b_002", np.array([5, 15], dtype=float), np.array([7.0, 6.5]), np.array([33.8, 33.9])),
        ],
    }

    def fake_read_argo_file(path, good_qc_only=True, min_levels=5):
        return profiles_by_file[str(Path(path))]

    monkeypatch.setattr("outlierdetect.parquet.read_argo_file", fake_read_argo_file)

    df = argo_directory_to_dataframe(root)
    assert len(df) == 7
    assert df.attrs["n_files"] == 2
    assert df.attrs["n_profiles"] == 3
    assert set(df["source_file"].unique()) == {str(root / "a.nc"), str(nested / "b.nc")}

    captured = {}

    def fake_to_parquet(self, path, index=False, engine="pyarrow"):
        captured["path"] = str(path)
        captured["index"] = index
        captured["engine"] = engine
        return None

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)
    output = tmp_path / "out" / "argo.parquet"
    summary = write_argo_parquet(root, output)
    assert summary.n_files == 2
    assert summary.n_profiles == 3
    assert summary.n_rows == 7
    assert captured["path"] == str(output)
    assert captured["index"] is False
    assert captured["engine"] == "pyarrow"


def test_argo_parquet_export_rejects_directory_output(tmp_path):
    from outlierdetect.parquet import write_argo_parquet

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    try:
        write_argo_parquet(tmp_path, output_dir)
    except IsADirectoryError as exc:
        assert "parquet file" in str(exc)
    else:
        raise AssertionError("Expected write_argo_parquet to reject directory outputs")


def test_argo_parquet_profiles_can_be_rehydrated(tmp_path, monkeypatch):
    import pandas as pd

    from outlierdetect.parquet import iter_argo_parquet_profiles

    df = pd.DataFrame(
        [
            {
                "source_file": "a.nc",
                "profile_index_in_file": 0,
                "profile_id": "a_001",
                "float_wmo": "5900001",
                "cycle_number": 1,
                "juld": 12345.0,
                "n_levels": 2,
                "level_index": 0,
                "pressure": 0.0,
                "temperature": 5.0,
                "salinity": 34.0,
            },
            {
                "source_file": "a.nc",
                "profile_index_in_file": 0,
                "profile_id": "a_001",
                "float_wmo": "5900001",
                "cycle_number": 1,
                "juld": 12345.0,
                "n_levels": 2,
                "level_index": 1,
                "pressure": 10.0,
                "temperature": 4.5,
                "salinity": 34.1,
            },
            {
                "source_file": "b.nc",
                "profile_index_in_file": 0,
                "profile_id": "b_001",
                "float_wmo": "5900002",
                "cycle_number": 2,
                "juld": 12346.0,
                "n_levels": 2,
                "level_index": 0,
                "pressure": 5.0,
                "temperature": 6.0,
                "salinity": 33.9,
            },
            {
                "source_file": "b.nc",
                "profile_index_in_file": 0,
                "profile_id": "b_001",
                "float_wmo": "5900002",
                "cycle_number": 2,
                "juld": 12346.0,
                "n_levels": 2,
                "level_index": 1,
                "pressure": 15.0,
                "temperature": 5.5,
                "salinity": 34.0,
            },
        ]
    )

    monkeypatch.setattr(pd, "read_parquet", lambda path: df)

    profiles = iter_argo_parquet_profiles(tmp_path / "dataset.parquet", min_levels=2)
    assert len(profiles) == 2
    assert profiles[0].profile_id == "a_001"
    assert np.allclose(profiles[0].pressure, [0.0, 10.0])
    assert profiles[0].float_wmo == "5900001"
    assert profiles[1].profile_id == "b_001"
    assert np.allclose(profiles[1].temperature, [6.0, 5.5])


def test_outlier_detect_raw_to_parquet_cli_dispatches(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from outlierdetect import cli

    captured = {}

    def fake_write_argo_parquet(root, output, *, pattern="**/*.nc", good_qc_only=True, min_levels=5):
        captured["root"] = root
        captured["output"] = output
        captured["pattern"] = pattern
        captured["good_qc_only"] = good_qc_only
        captured["min_levels"] = min_levels
        return SimpleNamespace(
            as_dict=lambda: {
                "output": str(output),
                "n_files": 1,
                "n_profiles": 1,
                "n_rows": 2,
            }
        )

    monkeypatch.setattr("outlierdetect.parquet.write_argo_parquet", fake_write_argo_parquet)

    input_dir = tmp_path / "argo"
    output = tmp_path / "out" / "argo.parquet"
    cli.main(
        [
            "--raw-to-parquet",
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--pattern",
            "**/profiles/*.nc",
            "--min-levels",
            "7",
            "--no-good-qc-only",
        ]
    )

    assert captured["root"] == str(input_dir)
    assert captured["output"] == output
    assert captured["pattern"] == "**/profiles/*.nc"
    assert captured["good_qc_only"] is False
    assert captured["min_levels"] == 7


def test_build_argo_synthetic_examples_accepts_parquet_path(tmp_path, monkeypatch):
    import pandas as pd

    from outlierdetect.training.argo import build_argo_synthetic_examples

    df = pd.DataFrame(
        [
            {
                "source_file": "a.nc",
                "profile_index_in_file": 0,
                "profile_id": "a_001",
                "float_wmo": "5900001",
                "cycle_number": 1,
                "juld": 12345.0,
                "n_levels": 5,
                "level_index": 0,
                "pressure": 0.0,
                "temperature": 5.0,
                "salinity": 34.0,
            },
            {
                "source_file": "a.nc",
                "profile_index_in_file": 0,
                "profile_id": "a_001",
                "float_wmo": "5900001",
                "cycle_number": 1,
                "juld": 12345.0,
                "n_levels": 5,
                "level_index": 1,
                "pressure": 10.0,
                "temperature": 4.8,
                "salinity": 34.05,
            },
            {
                "source_file": "a.nc",
                "profile_index_in_file": 0,
                "profile_id": "a_001",
                "float_wmo": "5900001",
                "cycle_number": 1,
                "juld": 12345.0,
                "n_levels": 5,
                "level_index": 2,
                "pressure": 20.0,
                "temperature": 4.5,
                "salinity": 34.1,
            },
            {
                "source_file": "a.nc",
                "profile_index_in_file": 0,
                "profile_id": "a_001",
                "float_wmo": "5900001",
                "cycle_number": 1,
                "juld": 12345.0,
                "n_levels": 5,
                "level_index": 3,
                "pressure": 30.0,
                "temperature": 4.2,
                "salinity": 34.15,
            },
            {
                "source_file": "a.nc",
                "profile_index_in_file": 0,
                "profile_id": "a_001",
                "float_wmo": "5900001",
                "cycle_number": 1,
                "juld": 12345.0,
                "n_levels": 5,
                "level_index": 4,
                "pressure": 40.0,
                "temperature": 4.0,
                "salinity": 34.2,
            },
        ]
    )

    monkeypatch.setattr(pd, "read_parquet", lambda path: df)

    synth = build_argo_synthetic_examples(
        str(tmp_path / "dataset.parquet"),
        n_examples_per_profile=1,
        n_levels=5,
        grid_size=8,
        seed=2,
        min_levels=5,
    )

    assert len(synth) == 1
    assert synth[0].example.profile.profile_id == "a_001"
    assert synth[0].example.profile.pressure.size == 5


def test_predict_loader_accepts_parquet_path(monkeypatch, tmp_path):
    from outlierdetect import cli
    from outlierdetect.argo import ArgoProfile

    captured = {}

    def fake_iter_parquet(path, min_levels=5):
        captured["path"] = path
        captured["min_levels"] = min_levels
        return [
            ArgoProfile(
                profile_id="pred_001",
                pressure=np.array([0.0, 10.0, 20.0], dtype=float),
                temperature=np.array([5.0, 4.7, 4.4], dtype=float),
                salinity=np.array([34.0, 34.1, 34.2], dtype=float),
            )
        ]

    monkeypatch.setattr(cli, "iter_argo_parquet_profiles", fake_iter_parquet)
    monkeypatch.setattr(cli, "iter_argo_files", lambda *args, **kwargs: [])

    profiles = cli._load_prediction_profiles(tmp_path / "profiles.parquet", min_levels=7)

    assert captured["path"] == tmp_path / "profiles.parquet"
    assert captured["min_levels"] == 7
    assert len(profiles) == 1


def test_training_run_writer_updates_progress_each_epoch(tmp_path, monkeypatch):
    import torch

    from outlierdetect.training.artifacts import TrainingRunWriter

    synth = degrade_highres_profile(
        np.linspace(0, 500, 60),
        5.0 - 3.0 * np.exp(-np.linspace(0, 500, 60) / 160.0),
        34.1 + 0.4 * (1.0 - np.exp(-np.linspace(0, 500, 60) / 200.0)),
        rng=np.random.default_rng(3),
        profile_id="run_writer_profile",
    )
    examples = [synth.example, synth.example]

    class DummyModel(torch.nn.Module):
        def forward(self, features, mask=None, recon_pressure=None):
            batch = int(features.shape[0])
            grid = 12
            profile = torch.zeros((batch,), dtype=features.dtype, device=features.device)
            point = torch.zeros((batch, grid, 3), dtype=features.dtype, device=features.device)
            recon = torch.zeros((batch, grid, 2), dtype=features.dtype, device=features.device)
            return {"profile_logit": profile, "point_logits": point, "recon_mean": recon}

    def fake_plot(self, path, **kwargs):
        path.write_text("plot", encoding="utf-8")

    monkeypatch.setattr("outlierdetect.training.artifacts.TrainingRunWriter._save_reconstruction_plot", fake_plot)

    writer = TrainingRunWriter(tmp_path, examples, plot_count=1, seed=5)
    history = [
        {"train_total": 1.5, "train_reconstruction": 0.7, "epoch": 1.0},
        {"val_total": 1.2, "val_reconstruction": 0.6, "epoch": 1.0},
    ]
    writer.record_epoch(
        epoch=1,
        model=DummyModel(),
        history=history,
        device="cpu",
        n_train_examples=1,
        n_val_examples=1,
    )

    progress_path = writer.progress_path
    assert progress_path.exists()
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["current_epoch"] == 1
    assert payload["epochs"][0]["train"]["total"] == 1.5
    assert payload["epochs"][0]["val"]["reconstruction"] == 0.6
    assert len(payload["plot_files"]) == 1
    assert (writer.run_dir / payload["plot_files"][0]).exists()
    assert len(payload["selected_profiles"]) == 1
    prediction_path = writer.run_dir / payload["selected_profiles"][0]["prediction_file"]
    assert prediction_path.exists()
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert np.isclose(prediction["profile_bad_probability"], 0.5)
    assert len(prediction["point_bad_t"]) == 12
    assert len(prediction["point_bad_s"]) == 12
    assert len(prediction["point_density_inconsistent"]) == 12

    final = writer.finalize(history=history, checkpoint_path=tmp_path / "checkpoint.pt")
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["checkpoint_path"].endswith("checkpoint.pt")
    assert final["status"] == "complete"


def test_train_main_writes_epoch_progress(tmp_path, monkeypatch):
    from outlierdetect import cli
    from outlierdetect.training.synthetic import degrade_highres_profile

    synth = degrade_highres_profile(
        np.linspace(0, 500, 60),
        5.0 - 3.0 * np.exp(-np.linspace(0, 500, 60) / 160.0),
        34.1 + 0.4 * (1.0 - np.exp(-np.linspace(0, 500, 60) / 200.0)),
        rng=np.random.default_rng(11),
        profile_id="train_main_profile",
    )
    synthetic = [synth]

    def fake_build_argo_synthetic_examples(*args, **kwargs):
        return synthetic

    def fake_fit_model(model, train_loader, val_loader=None, *, device="cpu", weights=None, grad_clip=1.0, epochs=1, optimizer=None, learning_rate=1e-3, epoch_callback=None):
        history = [{"train_total": 1.0, "epoch": 1.0}]
        if epoch_callback is not None:
            epoch_callback(1, model, history)
        return history

    def fake_plot(self, path, **kwargs):
        path.write_text("plot", encoding="utf-8")

    monkeypatch.setattr(cli, "build_argo_synthetic_examples", fake_build_argo_synthetic_examples)
    monkeypatch.setattr(cli, "fit_model", fake_fit_model)
    monkeypatch.setattr("outlierdetect.training.artifacts.TrainingRunWriter._save_reconstruction_plot", fake_plot)

    cli.train_main(
        [
            "--argo-root",
            str(tmp_path / "argo"),
            "--run-root",
            str(tmp_path / "runs"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--val-fraction",
            "0",
            "--device",
            "cpu",
        ]
    )

    progress_files = list((tmp_path / "runs").rglob("progress.json"))
    assert len(progress_files) == 1
    payload = json.loads(progress_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["current_epoch"] == 1
    assert len(payload["plot_files"]) == 1
    assert len(payload["selected_profiles"]) == 1
    prediction_path = progress_files[0].parent / payload["selected_profiles"][0]["prediction_file"]
    assert prediction_path.exists()


def test_predict_main_writes_probability_json(tmp_path, monkeypatch):
    from outlierdetect import cli
    from outlierdetect.argo import ArgoProfile
    import torch

    profiles = [
        ArgoProfile(
            profile_id="pred_001",
            pressure=np.array([0.0, 10.0, 30.0], dtype=float),
            temperature=np.array([5.0, 4.6, 4.1], dtype=float),
            salinity=np.array([34.0, 34.1, 34.2], dtype=float),
            cycle_number=1,
            float_wmo="5900001",
            juld=12345.0,
        ),
        ArgoProfile(
            profile_id="pred_002",
            pressure=np.array([0.0, 15.0, 45.0], dtype=float),
            temperature=np.array([4.8, 4.3, 3.8], dtype=float),
            salinity=np.array([34.05, 34.12, 34.18], dtype=float),
            cycle_number=2,
            float_wmo="5900002",
            juld=12346.0,
        ),
    ]

    class DummyModel(torch.nn.Module):
        def forward(self, features, mask=None, recon_pressure=None):
            batch = int(features.shape[0])
            levels = int(features.shape[1])
            grid = int(recon_pressure.shape[1]) if recon_pressure is not None else 80
            profile = torch.zeros((batch,), dtype=features.dtype, device=features.device)
            point = torch.zeros((batch, levels, 3), dtype=features.dtype, device=features.device)
            nuisance_mean = torch.zeros((batch, 4), dtype=features.dtype, device=features.device)
            nuisance_log_std = torch.zeros((batch, 4), dtype=features.dtype, device=features.device)
            recon = torch.zeros((batch, grid, 2), dtype=features.dtype, device=features.device)
            recon_log_std = torch.zeros((batch, grid, 2), dtype=features.dtype, device=features.device)
            return {
                "profile_logit": profile,
                "point_logits": point,
                "nuisance_mean": nuisance_mean,
                "nuisance_log_std": nuisance_log_std,
                "recon_mean": recon,
                "recon_log_std": recon_log_std,
            }

    def fake_load_model_from_checkpoint(path, model_factory, *, map_location="cpu"):
        return DummyModel(), {
            "input_dim": 34,
            "grid_size": 80,
            "normalization": None,
            "feature_names": [f"f{i}" for i in range(34)],
        }

    captured = {}

    def fake_iter_argo_files(*args, **kwargs):
        captured["good_qc_only"] = kwargs.get("good_qc_only")
        return profiles

    monkeypatch.setattr(cli, "iter_argo_files", fake_iter_argo_files)
    monkeypatch.setattr(cli, "load_model_from_checkpoint", fake_load_model_from_checkpoint)

    run_root = tmp_path / "predict-runs"
    cli.main(
        [
            "--predict",
            "--argo-root",
            str(tmp_path / "argo"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--run-root",
            str(run_root),
            "--device",
            "cpu",
        ]
    )

    progress_files = list(run_root.rglob("progress.json"))
    assert len(progress_files) == 1
    payload = json.loads(progress_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["mode"] == "predict"
    assert payload["n_profiles"] == 2
    assert payload["n_predicted"] == 2
    assert payload["n_failed"] == 0
    assert payload["plot_count"] == 2
    assert len(payload["prediction_files"]) == 2
    assert len(payload["plot_files"]) == 2
    assert captured["good_qc_only"] is False

    prediction_dir = progress_files[0].parent / "predictions"
    plot_dir = progress_files[0].parent / "plots" / "predict"
    assert len(list(prediction_dir.glob("*.json"))) == 2
    assert len(list(plot_dir.glob("*.png"))) == 2
    for prediction_file in payload["prediction_files"]:
        prediction_path = progress_files[0].parent / prediction_file
        assert prediction_path.exists()
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        assert np.isclose(prediction["profile_bad_probability"], 0.5)
        assert len(prediction["point_bad_t"]) == 3
        assert len(prediction["point_bad_s"]) == 3
        assert len(prediction["point_density_inconsistent"]) == 3


def test_predict_main_can_apply_good_qc_only(tmp_path, monkeypatch):
    from outlierdetect import cli
    from outlierdetect.argo import ArgoProfile
    import torch

    profiles = [
        ArgoProfile(
            profile_id="pred_002",
            pressure=np.array([0.0, 10.0, 30.0], dtype=float),
            temperature=np.array([5.0, 4.6, 4.1], dtype=float),
            salinity=np.array([34.0, 34.1, 34.2], dtype=float),
            cycle_number=2,
            float_wmo="5900002",
            juld=12346.0,
        )
    ]

    class DummyModel(torch.nn.Module):
        def forward(self, features, mask=None, recon_pressure=None):
            batch = int(features.shape[0])
            levels = int(features.shape[1])
            grid = int(recon_pressure.shape[1]) if recon_pressure is not None else 80
            profile = torch.zeros((batch,), dtype=features.dtype, device=features.device)
            point = torch.zeros((batch, levels, 3), dtype=features.dtype, device=features.device)
            nuisance_mean = torch.zeros((batch, 4), dtype=features.dtype, device=features.device)
            nuisance_log_std = torch.zeros((batch, 4), dtype=features.dtype, device=features.device)
            recon = torch.zeros((batch, grid, 2), dtype=features.dtype, device=features.device)
            recon_log_std = torch.zeros((batch, grid, 2), dtype=features.dtype, device=features.device)
            return {
                "profile_logit": profile,
                "point_logits": point,
                "nuisance_mean": nuisance_mean,
                "nuisance_log_std": nuisance_log_std,
                "recon_mean": recon,
                "recon_log_std": recon_log_std,
            }

    def fake_load_model_from_checkpoint(path, model_factory, *, map_location="cpu"):
        return DummyModel(), {
            "input_dim": 34,
            "grid_size": 80,
            "normalization": None,
            "feature_names": [f"f{i}" for i in range(34)],
        }

    captured = {}

    def fake_iter_argo_files(*args, **kwargs):
        captured["good_qc_only"] = kwargs.get("good_qc_only")
        return profiles

    monkeypatch.setattr(cli, "iter_argo_files", fake_iter_argo_files)
    monkeypatch.setattr(cli, "load_model_from_checkpoint", fake_load_model_from_checkpoint)

    run_root = tmp_path / "predict-runs"
    cli.main(
        [
            "--predict",
            "--argo-root",
            str(tmp_path / "argo"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--run-root",
            str(run_root),
            "--device",
            "cpu",
            "--good-qc-only",
        ]
    )

    assert captured["good_qc_only"] is True
    progress_files = list(run_root.rglob("progress.json"))
    assert len(progress_files) == 1
    payload = json.loads(progress_files[0].read_text(encoding="utf-8"))
    assert payload["plot_count"] == 1
    assert len(payload["plot_files"]) == 1
