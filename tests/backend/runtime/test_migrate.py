import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import migrate as migration
from backend.migrate import migration_files
from tests.support.paths import REPOSITORY_ROOT


def test_migration_files_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        migration_files(tmp_path / "missing")


def test_migration_files_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="contains no up migrations"):
        migration_files(tmp_path)


def test_migration_files_discovers_only_sorted_up_migrations(tmp_path: Path) -> None:
    for name in (
        "002_second.up.sql",
        "001_first.up.sql",
        "001_first.down.sql",
        "notes.sql",
    ):
        (tmp_path / name).write_text("SELECT 1;", encoding="utf-8")

    assert [path.name for path in migration_files(tmp_path)] == [
        "001_first.up.sql",
        "002_second.up.sql",
    ]


class MigrationCursor:
    def __init__(self, checksum: str | None = None) -> None:
        self.checksum = checksum

    def fetchone(self) -> tuple[str] | None:
        return (self.checksum,) if self.checksum is not None else None


class MigrationConnection:
    def __init__(self, applied: dict[str, str]) -> None:
        self.applied = dict(applied)
        self.executed: list[str] = []

    def execute(self, sql: str, parameters: tuple[str, ...] = ()) -> MigrationCursor:
        if sql.startswith("SELECT checksum"):
            return MigrationCursor(self.applied.get(parameters[0]))
        if sql.startswith("INSERT INTO schema_migration"):
            self.applied[parameters[0]] = parameters[1]
        elif sql != migration.CREATE_MIGRATION_TABLE:
            self.executed.append(sql)
        return MigrationCursor()

    def transaction(self) -> nullcontext[None]:
        return nullcontext()


@pytest.mark.parametrize("source", ["clean", "delivery", "upstream"])
def test_integrated_upgrade_preserves_full_migration_identities(
    source: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = REPOSITORY_ROOT / "migrations"
    files = migration_files(directory)
    delivery = {"009_org_units", "010_application_attribution"}
    checksums = {
        path.name.removesuffix(".up.sql"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    existing = {
        version: checksum for version, checksum in checksums.items()
        if (source == "delivery" and (version[:3] <= "008" or version in delivery))
        or (source == "upstream" and version not in delivery)
    }
    connection = MigrationConnection(existing)
    monkeypatch.setattr(
        migration, "get_settings",
        lambda: SimpleNamespace(
            database_url="unused-recording-connection", migrations_dir=directory
        ),
    )
    monkeypatch.setattr("backend.migrate.psycopg.connect", lambda _: nullcontext(connection))

    assert migration.migrate() == [version for version in checksums if version not in existing]
    assert connection.applied == checksums
    assert len(connection.executed) == len(checksums) - len(existing)
    assert migration.migrate() == []
    assert all(connection.applied[version] == checksum for version, checksum in existing.items())


def test_delivered_migration_checksums_are_unchanged() -> None:
    directory = REPOSITORY_ROOT / "migrations"
    expected = {
        "009_org_units": "f5ea92f081eb6c0c92c38b557be613160da1b2ccf3a551c400ac1ac70723d0ec",
        "010_application_attribution":
            "41533c4336664c15dee900de661ae54dd95b00be7650087ac5c182d838270124",
    }
    for version, checksum in expected.items():
        actual = hashlib.sha256((directory / f"{version}.up.sql").read_bytes()).hexdigest()
        assert actual == checksum