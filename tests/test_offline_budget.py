"""Without the tiktoken encoding (offline), budgets still hold."""

from __future__ import annotations

import pytest

from contexta_app import renderer
from contexta_app.graph import ProjectGraph
from contexta_app.renderer import estimate_tokens


@pytest.fixture
def offline_estimator(monkeypatch):
    # tiktoken downloads cl100k_base on first use; offline it falls back to len/4.
    monkeypatch.setattr(renderer, "_TOKEN_ENCODER", False)


def test_fallback_rounds_up(offline_estimator):
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    # Summing per-line estimates never undercounts the whole text.
    lines = ["  def f(a)  :1\n", "  def g()  :5\n", "pkg/m.py\n"]
    assert sum(estimate_tokens(line) for line in lines) >= estimate_tokens("".join(lines))


@pytest.mark.parametrize("budget", [100, 200, 400])
def test_repo_map_stays_inside_its_budget_offline(tmp_path, offline_estimator, budget):
    for i in range(30):
        body = "\n".join(f"def f{j}(a):\n    return m{(i + 1) % 30}.f{j}(a)\n" for j in range(6))
        (tmp_path / f"m{i}.py").write_text(body, encoding="utf-8")
    g = ProjectGraph(tmp_path)
    try:
        g.update(use_git=False)
        text = g.repo_map(budget)["text"]
        assert text
        assert estimate_tokens(text) <= budget
    finally:
        g.close()
