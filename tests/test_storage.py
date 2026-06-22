from youtube_toolkit.storage import DATABASE_FILENAME, STORAGE_DIRNAME, find_repo_root, resolve_storage_dir


def test_default_storage_path_is_repo_cache(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STORAGE_HOME", raising=False)
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STATE_HOME", raising=False)
    monkeypatch.delenv("YOUTUBE_MEMORY_HOME", raising=False)

    assert STORAGE_DIRNAME == ".cache"
    assert DATABASE_FILENAME == "youtube.sqlite3"
    assert resolve_storage_dir() == repo_root / ".cache"


def test_find_repo_root_returns_nearest_git_parent(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert find_repo_root(nested) == repo_root
