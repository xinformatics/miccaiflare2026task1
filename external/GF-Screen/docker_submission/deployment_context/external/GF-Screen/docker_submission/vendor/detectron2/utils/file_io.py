"""Local-only PathManager subset used by the imported FocalNet module."""

from pathlib import Path


class _PathManager:
    @staticmethod
    def open(path, mode="r", **kwargs):
        return Path(path).open(mode, **kwargs)

    @staticmethod
    def get_local_path(path, **_kwargs):
        return str(Path(path))


PathManager = _PathManager()

