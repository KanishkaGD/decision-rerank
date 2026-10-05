import importlib


def test_save_and_replace(tmp_path, monkeypatch):
    monkeypatch.setenv("CDM_RESULTS", str(tmp_path))
    import cdm.config
    import cdm.results_csv

    importlib.reload(cdm.config)
    rc = importlib.reload(cdm.results_csv)
    key = {"job": "rerank", "system": "s", "dataset": "d", "phrasing": "none"}
    assert not rc.result_exists(key)
    rc.save_result(key, {"nDCG@10": 0.5})
    assert rc.result_exists(key)
    rc.save_result(key, {"nDCG@10": 0.6})
    rc.save_result({**key, "dataset": "e"}, {"nDCG@10": 0.7, "ECE": 0.1})
    df = rc.load_results()
    assert len(df) == 2
    assert df[df["dataset"] == "d"]["nDCG@10"].tolist() == ["0.6"]
    assert list(df.columns[: len(rc.RESULT_KEY_COLUMNS)]) == rc.RESULT_KEY_COLUMNS


def test_unknown_key_column_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CDM_RESULTS", str(tmp_path))
    import cdm.results_csv as rc

    try:
        rc.save_result({"jobb": "x"}, {})
    except ValueError:
        return
    raise AssertionError("expected ValueError")
