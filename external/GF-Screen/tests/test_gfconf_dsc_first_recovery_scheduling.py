from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_gfconf_dsc_first_recovery_public_20260824.sh"
).read_text(encoding="utf-8")


def test_runner_is_public_only_and_cannot_package_or_upload():
    assert "Validation-Public-Labels" in SCRIPT
    assert "Validation-Hidden" not in SCRIPT
    assert "package_codabench" not in SCRIPT
    assert "codabench" not in SCRIPT.lower()
    assert "curl " not in SCRIPT and "docker push" not in SCRIPT


def test_runner_uses_immutable_anchor_and_dsc_first_gate():
    assert "DEPLOYMENT_CHAMPION.json" in SCRIPT
    assert "materialize_immutable_anchor_additions.py" in SCRIPT
    assert "select_dsc_first_nonregression.py" in SCRIPT
    assert "--min-mm3 1000" in SCRIPT
    assert "--min-nsd-gain 0.0" in SCRIPT


def test_runner_requires_bidirectional_model_agreement():
    assert "b32_by_focus" in SCRIPT
    assert "focus_by_b32" in SCRIPT
    assert "--support \"${support}\"" in SCRIPT
