from scripts.smoke_test import main


def test_smoke_script(capsys):
    main()

    output = capsys.readouterr().out
    assert "Smoke test passed for parkinson_wearable_biomarkers" in output
    assert "baseline.yaml" in output
