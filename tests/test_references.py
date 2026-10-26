from pathlib import Path

from agatesim.references import cited_ids, load_catalogue


def test_catalogue_loads(repo_root: Path) -> None:
    catalogue = load_catalogue(repo_root / "references" / "catalogue.toml")
    assert len(catalogue.references) > 0


def test_every_citation_resolves(repo_root: Path) -> None:
    known = load_catalogue(repo_root / "references" / "catalogue.toml").ids()
    docs = [repo_root / "README.md", *sorted((repo_root / "docs").rglob("*.md"))]
    unresolved: dict[str, set[str]] = {}
    for doc in docs:
        missing = cited_ids(doc.read_text("utf-8")) - known
        if missing:
            unresolved[str(doc.relative_to(repo_root))] = missing
    assert not unresolved, f"citations missing from references/catalogue.toml: {unresolved}"
