from argparse import Namespace
import json

import numpy as np
import pytest

from outlierdetect import Heuristic, ProfileInput, build_level_features, linear_detrend
from outlierdetect.argo import ArgoProfile
from outlierdetect.cli import _build_train_parser
from outlierdetect.en4 import read_en4_file
from outlierdetect.runtime_config import load_app_config, resolved_run_config_dict
from outlierdetect.training.artifacts import TrainingRunWriter
from outlierdetect.training.en4 import build_en4_synthetic_examples
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


def test_profile_features_use_gsw_density(monkeypatch):
    import sys
    import types

    fake_gsw = types.SimpleNamespace()

    def sa_from_sp(sp, pressure, lon, lat):
        sp = np.asarray(sp, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        return sp + 0.1 * float(lon) + 0.01 * float(lat) + 0.001 * pressure

    def ct_from_t(sa, temp, pressure):
        return np.asarray(temp, dtype=float) - 0.5

    def sigma0(sa, ct):
        return np.asarray(sa, dtype=float) - 2.0 * np.asarray(ct, dtype=float)

    fake_gsw.SA_from_SP = sa_from_sp
    fake_gsw.CT_from_t = ct_from_t
    fake_gsw.sigma0 = sigma0
    monkeypatch.setitem(sys.modules, "gsw", fake_gsw)

    profile = ProfileInput(
        pressure=np.array([0.0, 10.0, 20.0], dtype=float),
        temperature=np.array([5.0, 4.5, 4.0], dtype=float),
        salinity=np.array([34.0, 34.1, 34.3], dtype=float),
        attrs={"latitude": 10.0, "longitude": 20.0},
    )
    feats = build_level_features(profile)

    expected_density = (
        profile.salinity
        + 0.1 * 20.0
        + 0.01 * 10.0
        + 0.001 * profile.pressure
        - 2.0 * (profile.temperature - 0.5)
    )
    expected_pair = np.maximum(-(np.diff(expected_density) - 1e-4), 0.0)
    expected_level = np.zeros_like(profile.pressure)
    for idx, mag in enumerate(expected_pair):
        expected_level[idx] = max(expected_level[idx], mag)
        expected_level[idx + 1] = max(expected_level[idx + 1], mag)

    assert np.allclose(feats.column("density_proxy"), expected_density)
    assert np.allclose(feats.column("density_inversion_magnitude"), expected_level)
    assert feats.diagnostics["feature_version"].startswith("outlierdetect_v3_gsw_sigma0")


def test_profile_features_expose_bayesian_nuisance_posterior():
    p = np.array([0, 20, 60, 120, 200, 320], dtype=float)
    profile = ProfileInput(
        pressure=p,
        temperature=np.linspace(6.0, 2.0, p.size),
        salinity=np.linspace(34.1, 34.8, p.size),
        residual_t=np.array([0.10, 0.08, 0.04, 0.01, -0.02, -0.04], dtype=float),
        residual_s=np.array([0.02, 0.03, 0.05, 0.06, 0.07, 0.08], dtype=float),
        sigma_t=np.full(p.size, 0.25),
        sigma_s=np.full(p.size, 0.04),
        sigma_vert=np.full(p.size, 25.0),
        rho_ts=np.full(p.size, 0.35),
    )
    feats = build_level_features(profile)
    posterior = feats.diagnostics["correction_posterior"]

    assert "posterior_debiased_z_t" in feats.feature_names
    assert posterior.prior.std[2] > posterior.prior.std[0]
    assert posterior.prior.correlation[0, 2] > 0
    bias = posterior.as_nuisance_bias()
    assert "correlation" in bias.uncertainty
    assert bias.uncertainty["correlation"][0][2] > 0


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
    assert result.correction_posterior is not None
    assert "correlation" in result.nuisance_bias.uncertainty


def test_reconstruction_plot_colors_input_points_by_outlier_probability(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")

    import matplotlib
    from matplotlib.collections import PathCollection

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    writer = TrainingRunWriter(run_root=tmp_path / "runs", examples=[])

    captured: dict[str, object] = {}
    real_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    writer._save_reconstruction_plot(
        tmp_path / "plot.png",
        profile_id="demo",
        pressure=np.array([0.0, 10.0, 20.0]),
        temperature=np.array([5.0, 4.0, 3.0]),
        salinity=np.array([34.0, 34.2, 34.4]),
        point_outlier_probability=np.array([0.0, 0.5, 1.0]),
        recon_temperature=np.array([5.1, 4.1, 3.1]),
        recon_salinity=np.array([34.05, 34.15, 34.25]),
        truth_temperature=None,
        truth_salinity=None,
        epoch=1,
        rank=1,
    )

    fig = captured["fig"]
    ax = captured["ax"]
    scatter = next(
        coll
        for coll in ax.collections
        if isinstance(coll, PathCollection) and coll.get_cmap().name == "magma"
    )
    assert scatter.get_clim() == (0.0, 1.0)
    assert np.allclose(scatter.get_array(), np.array([0.0, 0.5, 1.0]))
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "point_outlier_probability"


def test_record_epoch_uses_model_density_probabilities(tmp_path, monkeypatch):
    pytest.importorskip("torch")

    import torch

    p_hr = np.linspace(0, 500, 120)
    t_hr = 5 - 4 * (1 - np.exp(-p_hr / 120))
    s_hr = 34.1 + 0.6 * (1 - np.exp(-p_hr / 180))
    synth = degrade_highres_profile(p_hr, t_hr, s_hr, rng=np.random.default_rng(2))
    example = synth.example

    writer = TrainingRunWriter(run_root=tmp_path / "runs", examples=[example], plot_count=1, seed=0)

    density_probs = torch.linspace(0.15, 0.85, example.profile.n_levels)
    t_s_prob = float(torch.sigmoid(torch.tensor(-1.0)).item())
    point_logits = torch.stack(
        [
            torch.full_like(density_probs, -1.0),
            torch.full_like(density_probs, -1.0),
            torch.logit(density_probs),
        ],
        dim=-1,
    ).unsqueeze(0)
    recon_mean = torch.stack(
        [
            torch.from_numpy(np.linspace(5.0, 3.0, 4)).float(),
            torch.from_numpy(np.linspace(34.0, 34.4, 4)).float(),
        ],
        dim=-1,
    ).unsqueeze(0)

    captured: dict[str, np.ndarray] = {}

    def capture_plot(self, *args, **kwargs):
        captured["probability"] = np.asarray(kwargs["point_outlier_probability"], dtype=float)

    class FakeModel:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def __call__(self, features, mask=None, recon_pressure=None):
            return {
                "point_logits": point_logits,
                "profile_logit": torch.tensor([0.0]),
                "recon_mean": recon_mean,
            }

    monkeypatch.setattr(TrainingRunWriter, "_save_reconstruction_plot", capture_plot)
    monkeypatch.setattr(TrainingRunWriter, "_write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(TrainingRunWriter, "_write_progress", lambda self: None)

    writer.record_epoch(epoch=1, model=FakeModel(), history=[], device="cpu")

    expected = np.maximum(density_probs.numpy(), t_s_prob)
    assert np.allclose(captured["probability"], expected)


def test_predict_cli_rejects_plot_count():
    from outlierdetect.cli import _build_predict_parser

    parser = _build_predict_parser()

    try:
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--predict-root",
                "data",
                "--plot-count",
                "3",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected --plot-count to be rejected by predict")


def test_predict_cli_rejects_legacy_argo_root_alias():
    from outlierdetect.cli import _build_predict_parser

    parser = _build_predict_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--checkpoint", "checkpoint.pt", "--argo-root", "data"])
    assert excinfo.value.code == 2


def test_en4_reader_applies_profile_and_level_qc(monkeypatch):
    class FakeVar:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getitem__(self, key):
            return self.data[key]

    class FakeDataset:
        def __init__(self, variables):
            self.variables = variables

        def __getitem__(self, key):
            return self.variables[key]

        def close(self):
            return None

    ds = FakeDataset(
        {
            "DEPH_CORRECTED": FakeVar(
                np.array([[0.0, 10.0, 20.0, 30.0], [0.0, 15.0, 25.0, 40.0]], dtype=float)
            ),
            "POTM_CORRECTED": FakeVar(
                np.array([[10.0, 9.5, 9.0, 8.5], [11.0, 10.5, 10.0, 9.5]], dtype=float)
            ),
            "PSAL_CORRECTED": FakeVar(
                np.array([[35.0, 34.9, 34.8, 34.7], [34.5, 34.6, 34.7, 34.8]], dtype=float)
            ),
            "DEPH_CORRECTED_QC": FakeVar(np.ones((2, 4), dtype=np.uint8)),
            "POTM_CORRECTED_QC": FakeVar(np.ones((2, 4), dtype=np.uint8)),
            "PSAL_CORRECTED_QC": FakeVar(np.array([[1, 1, 4, 1], [1, 1, 1, 1]], dtype=np.uint8)),
            "PROFILE_POTM_QC": FakeVar(np.array([1, 4], dtype=np.uint8)),
            "PROFILE_PSAL_QC": FakeVar(np.array([1, 1], dtype=np.uint8)),
            "CYCLE_NUMBER": FakeVar(np.array([7, 8], dtype=float)),
            "JULD": FakeVar(np.array([100.0, 130.0], dtype=float)),
            "LATITUDE": FakeVar(np.array([45.0, 46.0], dtype=float)),
            "LONGITUDE": FakeVar(np.array([-20.0, -21.0], dtype=float)),
        }
    )

    monkeypatch.setattr("outlierdetect.en4._open_nc", lambda path: ds)
    profiles = read_en4_file("demo_199001.nc", good_qc_only=True, min_levels=2)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.profile_id == "demo_199001_0007"
    assert profile.cycle_number == 7
    assert profile.latitude == 45.0
    assert profile.longitude == -20.0
    assert profile.n_levels == 3
    assert np.allclose(profile.pressure, np.array([0.0, 10.0, 30.0]))


def test_en4_reader_falls_back_when_primary_variable_is_unreadable(monkeypatch):
    class FakeVar:
        def __init__(self, data, *, fail_on_read: bool = False):
            self.data = np.asarray(data)
            self.fail_on_read = fail_on_read

        def __getitem__(self, key):
            if self.fail_on_read:
                raise RuntimeError("NetCDF: HDF error")
            return self.data[key]

    class FakeDataset:
        def __init__(self, variables):
            self.variables = variables

        def __getitem__(self, key):
            return self.variables[key]

        def close(self):
            return None

    ds = FakeDataset(
        {
            "DEPH_CORRECTED": FakeVar(np.array([[0.0, 10.0, 20.0]], dtype=float)),
            "POTM_CORRECTED": FakeVar(np.array([[99.0, 99.0, 99.0]], dtype=float), fail_on_read=True),
            "TEMP": FakeVar(np.array([[10.0, 9.5, 9.0]], dtype=float)),
            "PSAL_CORRECTED": FakeVar(np.array([[35.0, 34.9, 34.8]], dtype=float)),
            "CYCLE_NUMBER": FakeVar(np.array([7], dtype=float)),
            "JULD": FakeVar(np.array([100.0], dtype=float)),
            "LATITUDE": FakeVar(np.array([45.0], dtype=float)),
            "LONGITUDE": FakeVar(np.array([-20.0], dtype=float)),
        }
    )

    monkeypatch.setattr("outlierdetect.en4._open_nc", lambda path: ds)
    profiles = read_en4_file("demo_199001.nc", good_qc_only=False, min_levels=2)

    assert len(profiles) == 1
    profile = profiles[0]
    assert np.allclose(profile.temperature, np.array([10.0, 9.5, 9.0]))


def test_en4_reader_ignores_blank_profile_qc(monkeypatch):
    class FakeVar:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getitem__(self, key):
            return self.data[key]

    class FakeDataset:
        def __init__(self, variables):
            self.variables = variables

        def __getitem__(self, key):
            return self.variables[key]

        def close(self):
            return None

    ds = FakeDataset(
        {
            "DEPH_CORRECTED": FakeVar(np.array([[0.0, 10.0, 20.0]], dtype=float)),
            "TEMP": FakeVar(np.array([[10.0, 9.5, 9.0]], dtype=float)),
            "PSAL_CORRECTED": FakeVar(np.array([[35.0, 34.9, 34.8]], dtype=float)),
            "PROFILE_DEPH_QC": FakeVar(np.array([b""], dtype="S1")),
            "PROFILE_POTM_QC": FakeVar(np.array([1], dtype=np.uint8)),
            "PROFILE_PSAL_QC": FakeVar(np.array([1], dtype=np.uint8)),
            "CYCLE_NUMBER": FakeVar(np.array([7], dtype=float)),
            "JULD": FakeVar(np.array([100.0], dtype=float)),
            "LATITUDE": FakeVar(np.array([45.0], dtype=float)),
            "LONGITUDE": FakeVar(np.array([-20.0], dtype=float)),
        }
    )

    monkeypatch.setattr("outlierdetect.en4._open_nc", lambda path: ds)
    profiles = read_en4_file("demo_199001.nc", good_qc_only=True, min_levels=2)

    assert len(profiles) == 1
    assert np.allclose(profiles[0].temperature, np.array([10.0, 9.5, 9.0]))


def test_en4_reader_retries_without_qc_when_strict_qc_removes_all_profiles(monkeypatch):
    class FakeVar:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getitem__(self, key):
            return self.data[key]

    class FakeDataset:
        def __init__(self, variables):
            self.variables = variables

        def __getitem__(self, key):
            return self.variables[key]

        def close(self):
            return None

    ds = FakeDataset(
        {
            "DEPH_CORRECTED": FakeVar(np.array([[0.0, 10.0, 20.0]], dtype=float)),
            "POTM_CORRECTED": FakeVar(np.array([[10.0, 9.5, 9.0]], dtype=float)),
            "PSAL_CORRECTED": FakeVar(np.array([[35.0, 34.9, 34.8]], dtype=float)),
            "DEPH_CORRECTED_QC": FakeVar(np.ones((1, 3), dtype=np.uint8)),
            "POTM_CORRECTED_QC": FakeVar(np.ones((1, 3), dtype=np.uint8)),
            "PSAL_CORRECTED_QC": FakeVar(np.array([[4, 4, 4]], dtype=np.uint8)),
            "PROFILE_DEPH_QC": FakeVar(np.array([1], dtype=np.uint8)),
            "PROFILE_POTM_QC": FakeVar(np.array([1], dtype=np.uint8)),
            "PROFILE_PSAL_QC": FakeVar(np.array([1], dtype=np.uint8)),
            "CYCLE_NUMBER": FakeVar(np.array([7], dtype=float)),
            "JULD": FakeVar(np.array([100.0], dtype=float)),
            "LATITUDE": FakeVar(np.array([45.0], dtype=float)),
            "LONGITUDE": FakeVar(np.array([-20.0], dtype=float)),
        }
    )

    monkeypatch.setattr("outlierdetect.en4._open_nc", lambda path: ds)
    profiles = read_en4_file("demo_199001.nc", good_qc_only=True, min_levels=2)

    assert len(profiles) == 1
    assert np.allclose(profiles[0].salinity, np.array([35.0, 34.9, 34.8]))


def test_en4_reader_uses_raw_values_without_qc_masking(monkeypatch):
    class FakeVar:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getitem__(self, key):
            return self.data[key]

    class FakeDataset:
        def __init__(self, variables):
            self.variables = variables

        def __getitem__(self, key):
            return self.variables[key]

        def close(self):
            return None

    ds = FakeDataset(
        {
            "DEPH_CORRECTED": FakeVar(np.array([[0.0, 10.0, 20.0]], dtype=float)),
            "TEMP": FakeVar(np.array([[1.0, 2.0, 3.0]], dtype=float)),
            "PSAL": FakeVar(np.array([[31.0, 32.0, 33.0]], dtype=float)),
            "POTM_CORRECTED": FakeVar(np.array([[10.0, 20.0, 30.0]], dtype=float)),
            "PSAL_CORRECTED": FakeVar(np.array([[35.0, 36.0, 37.0]], dtype=float)),
            "PROFILE_DEPH_QC": FakeVar(np.array([1], dtype=np.uint8)),
            "PROFILE_POTM_QC": FakeVar(np.array([4], dtype=np.uint8)),
            "PROFILE_PSAL_QC": FakeVar(np.array([4], dtype=np.uint8)),
            "CYCLE_NUMBER": FakeVar(np.array([7], dtype=float)),
            "JULD": FakeVar(np.array([100.0], dtype=float)),
            "LATITUDE": FakeVar(np.array([45.0], dtype=float)),
            "LONGITUDE": FakeVar(np.array([-20.0], dtype=float)),
        }
    )

    monkeypatch.setattr("outlierdetect.en4._open_nc", lambda path: ds)
    profiles = read_en4_file("demo_199001.nc", good_qc_only=True, min_levels=2, use_raw_values=True)

    assert len(profiles) == 1
    profile = profiles[0]
    assert np.allclose(profile.temperature, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(profile.salinity, np.array([31.0, 32.0, 33.0]))


def test_argo_reader_uses_raw_values_without_qc_masking(monkeypatch):
    class FakeVar:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getitem__(self, key):
            return self.data[key]

    class FakeDataset:
        def __init__(self, variables):
            self.variables = variables

        def __getitem__(self, key):
            return self.variables[key]

        def close(self):
            return None

    ds = FakeDataset(
        {
            "PRES": FakeVar(np.array([[0.0, 10.0, 20.0]], dtype=float)),
            "TEMP": FakeVar(np.array([[1.0, 2.0, 3.0]], dtype=float)),
            "PSAL": FakeVar(np.array([[31.0, 32.0, 33.0]], dtype=float)),
            "TEMP_ADJUSTED": FakeVar(np.array([[10.0, 20.0, 30.0]], dtype=float)),
            "PSAL_ADJUSTED": FakeVar(np.array([[35.0, 36.0, 37.0]], dtype=float)),
            "PRES_QC": FakeVar(np.array([[4, 4, 4]], dtype=np.uint8)),
            "TEMP_ADJUSTED_QC": FakeVar(np.array([[4, 4, 4]], dtype=np.uint8)),
            "PSAL_ADJUSTED_QC": FakeVar(np.array([[4, 4, 4]], dtype=np.uint8)),
            "CYCLE_NUMBER": FakeVar(np.array([7], dtype=float)),
            "JULD": FakeVar(np.array([100.0], dtype=float)),
            "LATITUDE": FakeVar(np.array([45.0], dtype=float)),
            "LONGITUDE": FakeVar(np.array([-20.0], dtype=float)),
        }
    )

    monkeypatch.setattr("outlierdetect.argo._open_nc", lambda path: ds)
    from outlierdetect.argo import read_argo_file

    profiles = read_argo_file("demo_199001.nc", good_qc_only=True, min_levels=2, use_raw_values=True)

    assert len(profiles) == 1
    profile = profiles[0]
    assert np.allclose(profile.temperature, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(profile.salinity, np.array([31.0, 32.0, 33.0]))


def test_netcdf_backend_pads_truncated_hdf5(tmp_path, monkeypatch):
    import sys
    import types

    from outlierdetect.netcdf_backend import _open_padded_truncated_hdf5

    path = tmp_path / "truncated.nc"
    path.write_bytes(b"1234567890")

    calls: list[tuple[str, str, int | None]] = []

    class FakeHandle:
        def close(self):
            return None

    def fake_dataset(path_arg, mode="r", memory=None):
        calls.append((path_arg, mode, None if memory is None else len(memory)))
        if memory is None:
            raise OSError("primary open failed")
        assert len(memory) == 16
        return FakeHandle()

    fake_netCDF4 = types.ModuleType("netCDF4")
    fake_netCDF4.Dataset = fake_dataset
    monkeypatch.setitem(sys.modules, "netCDF4", fake_netCDF4)

    handle = _open_padded_truncated_hdf5(
        path,
        OSError("Unable to synchronously open file (truncated file: eof = 10, sblock->base_addr = 0, stored_eof = 16)"),
    )

    assert handle is not None
    assert calls == [("inmemory.nc", "r", 16)]


def test_en4_training_builder_marks_source_metadata():
    profile = ArgoProfile(
        profile_id="demo_0001",
        pressure=np.array([0.0, 10.0, 20.0, 30.0, 45.0], dtype=float),
        temperature=np.array([10.0, 9.5, 9.0, 8.5, 8.0], dtype=float),
        salinity=np.array([35.0, 34.9, 34.8, 34.7, 34.6], dtype=float),
        cycle_number=1,
        juld=100.0,
        latitude=45.0,
        longitude=-20.0,
    )

    synthetic = build_en4_synthetic_examples([profile], seed=2, n_examples_per_profile=1)

    assert len(synthetic) == 1
    example = synthetic[0].example.profile
    assert example.attrs["source"] == "en4"
    assert example.attrs["en4_profile_id"] == "demo_0001"


def test_train_cli_accepts_en4_data_source():
    parser = _build_train_parser()
    args = parser.parse_args(
        [
            "--train-root",
            "C:\\data\\en4",
            "--data-source",
            "en4",
        ]
    )

    assert args.data_source == "en4"


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
                "latitude": 42.5,
                "longitude": -68.0,
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
                "latitude": 42.5,
                "longitude": -68.0,
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
                "latitude": 41.0,
                "longitude": -67.5,
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
                "latitude": 41.0,
                "longitude": -67.5,
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
    assert profiles[0].latitude == 42.5
    assert profiles[0].longitude == -68.0
    assert profiles[1].profile_id == "b_001"
    assert np.allclose(profiles[1].temperature, [6.0, 5.5])
    assert profiles[1].latitude == 41.0
    assert profiles[1].longitude == -67.5


def test_load_app_config_resolves_relative_paths(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "outlierdetect.toml"
    config_path.write_text(
        """
[paths]
data_root = "data"
model_checkpoint = "checkpoints/train_dataset_20ep.pt"
train_run_root = "runs/train"
predict_run_root = "runs/predict"

[heave]
source = "heave/sigma.nc"

[inputs]
residual_t = false
residual_s = true
sigma_t = false
sigma_s = true
sigma_vert = false
sigma_heave_t = true
sigma_heave_s = false
rho_ts = true
day_of_year = false

[train]
argo_root = "argo/train"
test_root = "argo/test"
output = "checkpoints/train.pt"
run_root = "runs/train"
epochs = 12
learning_rate = 0.005
val_fraction = 0.2

[predict]
predict_root = "argo/predict"
checkpoint = "checkpoints/predict.pt"
run_root = "runs/predict"
""",
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.paths.data_root == config_dir / "data"
    assert config.paths.model_checkpoint == config_dir / "checkpoints/train_dataset_20ep.pt"
    assert config.paths.train_run_root == config_dir / "runs/train"
    assert config.paths.predict_run_root == config_dir / "runs/predict"
    assert config.heave.source == config_dir / "heave/sigma.nc"
    assert config.train.train_root == config_dir / "argo/train"
    assert config.train.test_root == config_dir / "argo/test"
    assert config.train.output == config_dir / "checkpoints/train.pt"
    assert config.predict.predict_root == config_dir / "argo/predict"
    assert config.predict.checkpoint == config_dir / "checkpoints/predict.pt"


def test_resolved_run_config_dict_overrides_toml_defaults(tmp_path):
    config = load_app_config(None)
    args = Namespace(
        config=tmp_path / "outlierdetect.toml",
        train_root=tmp_path / "argo",
        checkpoint=tmp_path / "checkpoint.pt",
        output=tmp_path / "model.pt",
        seed=99,
        profile_limit=3,
        test_root=tmp_path / "argo-test",
        n_examples_per_profile=2,
        n_levels=21,
        min_levels=6,
        grid_size=64,
        upper_ocean_bias=1.9,
        epochs=14,
        batch_size=4,
        learning_rate=0.01,
        val_fraction=0.25,
        test_augment=False,
        device="cpu",
        run_root=tmp_path / "runs",
        epoch_plot_count=5,
        good_qc_only=False,
        pattern="**/*.nc",
        use_residual_t=False,
        use_residual_s=True,
        use_sigma_t=False,
        use_sigma_s=True,
        use_sigma_vert=False,
        use_sigma_heave_t=True,
        use_sigma_heave_s=False,
        use_rho_ts=True,
        use_day_of_year=False,
        sigma_heave_source=False,
    )

    resolved = resolved_run_config_dict(config=config, command="train", args=args)

    assert resolved["paths"]["data_root"] == str(tmp_path / "argo")
    assert resolved["paths"]["model_checkpoint"] == str(tmp_path / "model.pt")
    assert resolved["paths"]["train_run_root"] == str(tmp_path / "runs")
    assert resolved["train"]["train_root"] == str(tmp_path / "argo")
    assert resolved["train"]["test_root"] == str(tmp_path / "argo-test")
    assert resolved["train"]["epochs"] == 14
    assert resolved["train"]["learning_rate"] == 0.01
    assert resolved["train"]["run_root"] == str(tmp_path / "runs")
    assert resolved["inputs"]["residual_t"] is False
    assert resolved["inputs"]["sigma_vert"] is False
    assert resolved["heave"]["source"] is False
    assert resolved["runtime"]["command"] == "train"
    assert resolved["runtime"]["train_root"] == str(tmp_path / "argo")


def test_config_init_writes_starter_toml(tmp_path, capsys):
    from outlierdetect import cli

    output = tmp_path / "config" / "outlierdetect.toml"
    cli.main(["config", "init", "--output", str(output)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "written"
    assert payload["output"] == str(output)
    text = output.read_text(encoding="utf-8")
    assert "[paths]" in text
    assert "train_root = \"data\"" in text
    assert "source = false" in text
    assert "outlierdetect config init" in text


def test_config_show_prints_default_template(capsys):
    from outlierdetect import cli

    cli.main(["config", "show"])
    text = capsys.readouterr().out
    assert "[train]" in text
    assert "train_root = \"data\"" in text
    assert "outlierdetect config show" in text


def test_config_validate_prints_resolved_config(tmp_path, capsys):
    from outlierdetect import cli

    config_path = tmp_path / "outlierdetect.toml"
    config_path.write_text(
        """
[paths]
data_root = "data"
model_checkpoint = "checkpoints/train_dataset_20ep.pt"

[train]
argo_root = "argo/train"
output = "checkpoints/train.pt"

[predict]
predict_root = "argo/predict"
checkpoint = "checkpoints/predict.pt"
""",
        encoding="utf-8",
    )

    cli.main(["config", "validate", "--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid"
    assert payload["config_path"] == str(config_path.resolve())
    assert payload["config"]["paths"]["data_root"] == str(tmp_path / "data")
    assert payload["config"]["train"]["train_root"] == str(tmp_path / "argo/train")
    assert payload["config"]["train"]["output"] == str(tmp_path / "checkpoints/train.pt")
    assert payload["config"]["predict"]["predict_root"] == str(tmp_path / "argo/predict")
    assert payload["config"]["predict"]["checkpoint"] == str(tmp_path / "checkpoints/predict.pt")


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
            grid = 20
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
    assert len(prediction["point_bad_t"]) == 20
    assert len(prediction["point_bad_s"]) == 20
    assert len(prediction["point_density_inconsistent"]) == 20

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

    def fake_fit_model(
        model,
        train_loader,
        val_loader=None,
        *,
        eval_label="val",
        device="cpu",
        weights=None,
        grad_clip=1.0,
        epochs=1,
        optimizer=None,
        learning_rate=1e-3,
        epoch_callback=None,
    ):
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
            "--train-root",
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


def test_train_main_uses_test_root_without_augment(tmp_path, monkeypatch):
    from outlierdetect import cli
    from outlierdetect.argo import ArgoProfile
    from outlierdetect.training.synthetic import degrade_highres_profile

    train_root = tmp_path / "train"
    test_root = tmp_path / "test"

    train_synth = degrade_highres_profile(
        np.linspace(0, 500, 60),
        5.0 - 3.0 * np.exp(-np.linspace(0, 500, 60) / 160.0),
        34.1 + 0.4 * (1.0 - np.exp(-np.linspace(0, 500, 60) / 200.0)),
        rng=np.random.default_rng(11),
        profile_id="train_main_train_profile",
    )
    raw_test_profile = ArgoProfile(
        profile_id="train_main_raw_test_profile",
        pressure=np.array([0.0, 25.0, 75.0], dtype=float),
        temperature=np.array([4.9, 4.4, 3.9], dtype=float),
        salinity=np.array([34.02, 34.08, 34.15], dtype=float),
        cycle_number=2,
        float_wmo="5901234",
        juld=12347.0,
    )

    calls = []
    raw_calls = {}

    def fake_build_argo_synthetic_examples(root, **kwargs):
        calls.append(str(root))
        if str(root) == str(train_root):
            assert kwargs.get("use_raw_values") is False
        if str(root) == str(train_root):
            return [train_synth]
        raise AssertionError(f"Unexpected root: {root}")

    def fake_iter_argo_files(root, **kwargs):
        raw_calls["root"] = str(root)
        raw_calls["good_qc_only"] = kwargs.get("good_qc_only")
        raw_calls["use_raw_values"] = kwargs.get("use_raw_values")
        return [raw_test_profile]

    captured = {}

    def fake_fit_model(
        model,
        train_loader,
        val_loader=None,
        *,
        eval_label="val",
        device="cpu",
        weights=None,
        grad_clip=1.0,
        epochs=1,
        optimizer=None,
        learning_rate=1e-3,
        epoch_callback=None,
    ):
        captured["train_ids"] = [ex.profile.profile_id for ex in train_loader.dataset.examples]
        captured["val_loader_none"] = val_loader is None
        captured["eval_ids"] = [] if val_loader is None else [ex.profile.profile_id for ex in val_loader.dataset.examples]
        captured["eval_label"] = eval_label
        history = [{"train_total": 1.0, "epoch": 1.0}]
        if epoch_callback is not None:
            epoch_callback(1, model, history)
        return history

    def fake_plot(self, path, **kwargs):
        path.write_text("plot", encoding="utf-8")

    monkeypatch.setattr(cli, "build_argo_synthetic_examples", fake_build_argo_synthetic_examples)
    monkeypatch.setattr(cli, "iter_argo_files", fake_iter_argo_files)
    monkeypatch.setattr(cli, "fit_model", fake_fit_model)
    monkeypatch.setattr("outlierdetect.training.artifacts.TrainingRunWriter._save_reconstruction_plot", fake_plot)

    cli.train_main(
        [
            "--train-root",
            str(train_root),
            "--test-root",
            str(test_root),
            "--run-root",
            str(tmp_path / "runs"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--val-fraction",
            "0.25",
            "--device",
            "cpu",
        ]
    )

    assert calls == [str(train_root)]
    assert raw_calls["root"] == str(test_root)
    assert raw_calls["good_qc_only"] is False
    assert raw_calls["use_raw_values"] is True
    assert captured["train_ids"] == ["train_main_train_profile"]
    assert captured["val_loader_none"] is True
    assert captured["eval_ids"] == []
    assert captured["eval_label"] == "test"

    progress_files = list((tmp_path / "runs").rglob("progress.json"))
    assert len(progress_files) == 1
    payload = json.loads(progress_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["eval_label"] == "test"
    assert payload["n_test_examples"] == 1
    assert payload["epochs"][0]["train"]["total"] == 1.0
    assert "test" not in payload["epochs"][0]
    assert payload["selected_profiles"][0]["profile_id"] == "train_main_raw_test_profile"


def test_train_main_uses_test_root_with_augment(tmp_path, monkeypatch):
    from outlierdetect import cli
    from outlierdetect.training.synthetic import degrade_highres_profile

    train_root = tmp_path / "train"
    test_root = tmp_path / "test"

    train_synth = degrade_highres_profile(
        np.linspace(0, 500, 60),
        5.0 - 3.0 * np.exp(-np.linspace(0, 500, 60) / 160.0),
        34.1 + 0.4 * (1.0 - np.exp(-np.linspace(0, 500, 60) / 200.0)),
        rng=np.random.default_rng(11),
        profile_id="train_main_train_profile",
    )
    test_synth = degrade_highres_profile(
        np.linspace(0, 500, 60),
        4.8 - 2.8 * np.exp(-np.linspace(0, 500, 60) / 150.0),
        34.05 + 0.35 * (1.0 - np.exp(-np.linspace(0, 500, 60) / 190.0)),
        rng=np.random.default_rng(12),
        profile_id="train_main_test_profile",
    )

    calls = []

    def fake_build_argo_synthetic_examples(root, **kwargs):
        calls.append((str(root), kwargs.get("use_raw_values")))
        assert kwargs.get("use_raw_values") is False
        if str(root) == str(train_root):
            return [train_synth]
        if str(root) == str(test_root):
            return [test_synth]
        raise AssertionError(f"Unexpected root: {root}")

    captured = {}

    def fake_fit_model(
        model,
        train_loader,
        val_loader=None,
        *,
        eval_label="val",
        device="cpu",
        weights=None,
        grad_clip=1.0,
        epochs=1,
        optimizer=None,
        learning_rate=1e-3,
        epoch_callback=None,
    ):
        captured["train_ids"] = [ex.profile.profile_id for ex in train_loader.dataset.examples]
        captured["val_loader_none"] = val_loader is None
        captured["eval_ids"] = [] if val_loader is None else [ex.profile.profile_id for ex in val_loader.dataset.examples]
        captured["eval_label"] = eval_label
        history = [{"train_total": 1.0, "test_total": 0.5, "epoch": 1.0}]
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
            "--train-root",
            str(train_root),
            "--test-root",
            str(test_root),
            "--test-augment",
            "--run-root",
            str(tmp_path / "runs"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--val-fraction",
            "0.25",
            "--device",
            "cpu",
        ]
    )

    assert calls == [(str(train_root), False), (str(test_root), False)]
    assert captured["train_ids"] == ["train_main_train_profile"]
    assert captured["val_loader_none"] is False
    assert captured["eval_ids"] == ["train_main_test_profile"]
    assert captured["eval_label"] == "test"

    progress_files = list((tmp_path / "runs").rglob("progress.json"))
    assert len(progress_files) == 1
    payload = json.loads(progress_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["eval_label"] == "test"
    assert payload["n_test_examples"] == 1
    assert payload["epochs"][0]["test"]["total"] == 0.5


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
            "--predict-root",
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
            "--predict-root",
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
