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


def test_rename_system(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("CDM_RESULTS", str(tmp_path))
    import cdm.config
    import cdm.results_csv as rc

    importlib.reload(cdm.config)
    importlib.reload(rc)
    rc.save_result({"job": "rerank", "system": "a-old", "dataset": "dl21"}, {"nDCG@10": 0.5})
    rc.save_result({"job": "rerank", "system": "b", "dataset": "dl21"}, {"nDCG@10": 0.4})
    assert rc.rename_system("a-old", "a-new") == 1
    assert sorted(rc.load_results()["system"]) == ["a-new", "b"]
    rc.save_result({"job": "rerank", "system": "a-old", "dataset": "dl21"}, {"nDCG@10": 0.6})
    try:
        rc.rename_system("a-old", "a-new")
        raise AssertionError("collision not detected")
    except ValueError:
        pass
