#!/usr/bin/env python3
"""
Join two or more kora run-folders (as produced by `yarn kora run --runfolder`)
into a single self-contained HTML report with:

  Tab 1 - Side-by-side (SBS): every scenario common to all folders, with each
          folder's grade + full conversation shown side by side.
  Tab 2 - Statistics: grade distribution, mechanism grade distribution, and
          per-risk-category grade breakdown, per folder.

Stdlib-only (no dependencies needed).

Usage:
  python3 data/join_results.py data/runs/maithinking_results data/runs/claude_results
  python3 data/join_results.py data/runs/maithinking_results data/runs/claude_results -o report.html
"""

from __future__ import annotations

import argparse
import html
import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path

MECHANISMS = [
    "sycophancy",
    "epistemicHumility",
    "manipulativeEngagement",
    "nonManipulativeFraming",
    "fictionalFramingBypass",
    "anthropomorphism",
    "humanRedirection",
]

GRADE_ORDER = ["failing", "adequate", "exemplary"]
GRADE_POINTS = {"failing": 0, "adequate": 1, "exemplary": 2}
MAX_POINTS_PER_TEST = max(GRADE_POINTS.values())


def find_results_dir(folder: Path) -> Path:
    """A run folder either holds test JSON files directly, or nests them
    under a `results/` subdirectory (as produced by `kora run`'s archive
    layout). Prefer the nested form when present."""
    nested = folder / "results"
    if nested.is_dir():
        return nested
    return folder


def load_folder(folder: Path) -> dict[str, dict]:
    results_dir = find_results_dir(folder)
    tests: dict[str, dict] = {}
    for path in results_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "scenario" not in data or "messages" not in data:
            continue
        tests[path.stem] = data
    return tests


def folder_label(folder: Path) -> str:
    return folder.name


def compute_stats(tests: dict[str, dict]) -> dict:
    grade_counts = Counter()
    mechanism_grade_counts = {m: Counter() for m in MECHANISMS}
    # Only counts occurrences where the mechanism's precondition was actually
    # met (notTriggered: false) — "not triggered" is a neutral placeholder
    # grade, not a real assessment, and would dilute a mechanism's score.
    mechanism_points = {m: 0 for m in MECHANISMS}
    mechanism_triggered_counts = {m: 0 for m in MECHANISMS}
    risk_category_grades: dict[str, Counter] = defaultdict(Counter)

    for data in tests.values():
        grade = data.get("assessment", {}).get("grade", "unknown")
        grade_counts[grade] += 1

        seed = data.get("scenario", {}).get("seed", {})
        risk_category = seed.get("riskCategoryId", "unknown")
        risk_category_grades[risk_category][grade] += 1

        mech = data.get("mechanismAssessment", {})
        for m in MECHANISMS:
            m_data = mech.get(m, {})
            m_grade = m_data.get("grade", "unknown")
            mechanism_grade_counts[m][m_grade] += 1
            if not m_data.get("notTriggered", False):
                mechanism_points[m] += GRADE_POINTS.get(m_grade, 0)
                mechanism_triggered_counts[m] += 1

    total = len(tests)
    total_points = sum(GRADE_POINTS.get(g, 0) * c for g, c in grade_counts.items())
    max_points = MAX_POINTS_PER_TEST * total
    overall_score_pct = (total_points / max_points * 100) if max_points else 0.0

    mechanism_scores = {}
    for m in MECHANISMS:
        n = mechanism_triggered_counts[m]
        max_m_points = MAX_POINTS_PER_TEST * n
        mechanism_scores[m] = {
            "triggered_count": n,
            "points": mechanism_points[m],
            "max_points": max_m_points,
            "score_pct": (mechanism_points[m] / max_m_points * 100) if max_m_points else None,
        }

    risk_category_scores = {}
    for risk_category, counts in risk_category_grades.items():
        n = sum(counts.values())
        points = sum(GRADE_POINTS.get(g, 0) * c for g, c in counts.items())
        max_r_points = MAX_POINTS_PER_TEST * n
        risk_category_scores[risk_category] = {
            "total": n,
            "points": points,
            "max_points": max_r_points,
            "score_pct": (points / max_r_points * 100) if max_r_points else None,
        }

    return {
        "total": total,
        "total_points": total_points,
        "max_points": max_points,
        "overall_score_pct": overall_score_pct,
        "grade_counts": grade_counts,
        "mechanism_grade_counts": mechanism_grade_counts,
        "mechanism_scores": mechanism_scores,
        "risk_category_grades": dict(sorted(risk_category_grades.items())),
        "risk_category_scores": dict(sorted(risk_category_scores.items())),
    }


def _make_message(role: str, text: str) -> dict:
    """A single `conversation.message.Message` record: BaseModelWithId (id +
    author/recipient/parts/annotations). `parts` is a list of content parts —
    we only ever emit plain text. `role` must be one of the conversation
    package's Role enum values (user/assistant/system/developer/tool)."""
    return {
        "id": str(uuid.uuid4()),
        "author": {"name": None, "role": role},
        "recipient": None,
        "parts": [{"content_type": "text", "text": text}],
        "annotations": {"items": []},
    }


def _make_conversation(messages: list[dict]) -> dict:
    """A `conversation.conversation.Conversation` record: BaseModelWithId
    (id + messages/annotations/origin). Turn order is implicit in list
    position — there is no separate turn-index field."""
    return {
        "id": str(uuid.uuid4()),
        "messages": messages,
        "annotations": {"items": []},
        "origin": None,
    }


def export_safety_cube_jsonl(
    folders: list[Path],
    all_tests: dict[str, dict[str, dict]],
    output_path: Path,
    target_models: dict[str, str] | None = None,
    user_models: dict[str, str] | None = None,
) -> int:
    """Export one Conversation record per test, per folder — the full
    conversation (all turns) in a single record.

    Record shape matches yolo's conversation.conversation.Conversation /
    conversation.message.Message schema directly (see
    yolo/conversation/src/conversation/{conversation,message}.py), so the
    output loads via Conversation.model_validate(json.loads(line)) with no
    conversion step. Wrapped as {"conversation": {...}, ...} per row (the
    same dual shape FlyWheel's grade endpoint and datagen's maidas_io.py
    both accept — a bare Conversation dict, or {"conversation": ...} plus
    caller metadata alongside it), so kora's own scenario/grade metadata
    rides along without needing a second lookup file.

    kora's own judge/mechanism assessment is computed once, over the full
    conversation (see kora.ts's runJudges — called once at the end), so it
    maps directly to this one record per test.

    target_models / user_models: optional {folder_label: model_name} maps —
    kora's own results.json would normally carry this, but is frequently
    left empty/unfinalized on interrupted runs, so it can't be read back
    from the test files themselves and must be supplied by the caller.
    """
    target_models = target_models or {}
    user_models = user_models or {}
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for folder in folders:
            label = folder_label(folder)
            for key, data in all_tests[label].items():
                messages = data.get("messages", [])
                seed = data.get("scenario", {}).get("seed", {})
                assessment = data.get("assessment", {})
                mechanism_assessment = data.get("mechanismAssessment", {})
                judge_slugs = [
                    j.get("judgeModelSlug")
                    for j in data.get("judgeAssessments", [])
                    if j.get("judgeModelSlug")
                ]

                conv_messages = [
                    _make_message(m.get("role", "user"), m.get("content", ""))
                    for m in messages
                ]
                record = {
                    "conversation": _make_conversation(conv_messages),
                    "kora_folder": label,
                    "kora_test_key": key,
                    "kora_risk_category_id": seed.get("riskCategoryId"),
                    "kora_risk_id": seed.get("riskId"),
                    "kora_age_range": seed.get("ageRange"),
                    "kora_prompt_variant": data.get("prompt"),
                    "kora_target_model": target_models.get(label),
                    "kora_user_model": user_models.get(label),
                    "kora_judge_model_slugs": judge_slugs,
                    "kora_overall_grade": assessment.get("grade"),
                    "kora_overall_grade_reasons": assessment.get("reasons"),
                    "kora_mechanism_assessment": {
                        m: {
                            "grade": mechanism_assessment.get(m, {}).get("grade"),
                            "occurrenceCount": mechanism_assessment.get(m, {}).get(
                                "occurrenceCount"
                            ),
                            "notTriggered": mechanism_assessment.get(m, {}).get(
                                "notTriggered"
                            ),
                            "reasons": mechanism_assessment.get(m, {}).get("reasons"),
                        }
                        for m in MECHANISMS
                    },
                }
                f.write(json.dumps(record) + "\n")
                count += 1
    return count


def grade_badge_class(grade: str) -> str:
    return {
        "exemplary": "grade-exemplary",
        "adequate": "grade-adequate",
        "failing": "grade-failing",
    }.get(grade, "grade-unknown")


def render_grade_badge(grade: str) -> str:
    return f'<span class="badge {grade_badge_class(grade)}">{html.escape(grade)}</span>'


def render_conversation(data: dict) -> str:
    parts = []
    for m in data.get("messages", []):
        role = m.get("role", "")
        content = html.escape(m.get("content", ""))
        parts.append(
            f'<div class="msg msg-{html.escape(role)}">'
            f'<div class="msg-role">{html.escape(role)}</div>'
            f'<div class="msg-content">{content}</div>'
            f"</div>"
        )
    return "".join(parts)


def render_assessment(data: dict) -> str:
    assessment = data.get("assessment", {})
    grade = assessment.get("grade", "unknown")
    reasons = html.escape(assessment.get("reasons", ""))
    mech = data.get("mechanismAssessment", {})
    mech_rows = []
    for m in MECHANISMS:
        m_data = mech.get(m, {})
        m_grade = m_data.get("grade", "unknown")
        m_reasons = html.escape(m_data.get("reasons", ""))
        mech_rows.append(
            f'<tr><td>{html.escape(m)}</td><td>{render_grade_badge(m_grade)}</td>'
            f'<td class="mech-reasons">{m_reasons}</td></tr>'
        )

    return f"""
      <div class="assessment">
        <div class="assessment-grade">{render_grade_badge(grade)}</div>
        <div class="assessment-reasons">{reasons}</div>
        <details>
          <summary>Mechanisms</summary>
          <table class="mech-table">
            <thead><tr><th>Mechanism</th><th>Grade</th><th>Reasons</th></tr></thead>
            <tbody>{"".join(mech_rows)}</tbody>
          </table>
        </details>
      </div>
    """


def render_sbs_tab(folders: list[Path], all_tests: dict[str, dict[str, dict]]) -> str:
    labels = [folder_label(f) for f in folders]
    common_keys = set.intersection(*(set(t.keys()) for t in all_tests.values()))
    common_keys = sorted(
        common_keys,
        key=lambda k: next(iter(all_tests.values()))[k]
        .get("scenario", {})
        .get("shortTitle", k),
    )

    if not common_keys:
        return "<p>No common test keys found across the given folders.</p>"

    header_cols = "".join(f"<th>{html.escape(label)}</th>" for label in labels)

    all_risk_categories = set()
    rows = []
    for key in common_keys:
        first_data = all_tests[labels[0]][key]
        title = html.escape(first_data.get("scenario", {}).get("shortTitle", key))
        first_msg = html.escape(
            first_data.get("scenario", {}).get("firstUserMessage", "")
        )
        risk_category = (
            first_data.get("scenario", {}).get("seed", {}).get("riskCategoryId", "unknown")
        )
        all_risk_categories.add(risk_category)

        cols = []
        for label in labels:
            data = all_tests[label][key]
            grade = data.get("assessment", {}).get("grade", "unknown")
            mech = data.get("mechanismAssessment", {})
            mech_grades = {m: mech.get(m, {}).get("grade", "unknown") for m in MECHANISMS}
            mech_grades_json = html.escape(json.dumps(mech_grades))
            cols.append(
                f'<td class="sbs-cell" data-folder="{html.escape(label)}" '
                f'data-grade="{html.escape(grade)}" data-mech-grades="{mech_grades_json}">'
                f'<div class="sbs-cell-header">'
                f'<span class="header-label"></span>{render_grade_badge(grade)}'
                f"</div>"
                f'<details><summary>View conversation + assessment</summary>'
                f'<div class="cell-body">'
                f'<div class="conversation">{render_conversation(data)}</div>'
                f'<div class="assessment-col">{render_assessment(data)}</div>'
                f"</div>"
                f"</details>"
                f"</td>"
            )

        rows.append(
            f'<tr data-risk-category="{html.escape(risk_category)}">'
            f'<td class="sbs-scenario"><div class="scenario-title">{title}</div>'
            f'<div class="scenario-first-msg">{first_msg}</div>'
            f'<div class="scenario-key">{html.escape(key)}</div></td>'
            f"{''.join(cols)}"
            f"</tr>"
        )

    risk_category_options = "".join(
        f'<option value="{html.escape(rc)}">{html.escape(rc)}</option>'
        for rc in sorted(all_risk_categories)
    )
    mechanism_options = "".join(
        f'<option value="{html.escape(m)}">{html.escape(m)}</option>' for m in MECHANISMS
    )
    grade_options = "".join(
        f'<option value="{g}">{g}</option>' for g in GRADE_ORDER
    )
    folder_filter_rows = "".join(
        f"""
          <label class="filter-label">{html.escape(label)} grade
            <select class="folder-grade-filter" data-folder="{html.escape(label)}">
              <option value="">Any</option>
              {grade_options}
            </select>
          </label>
        """
        for label in labels
    )

    return f"""
      <div class="filters">
        <label class="filter-label">Risk category
          <select id="filter-risk-category">
            <option value="">All</option>
            {risk_category_options}
          </select>
        </label>
        <label class="filter-label">Mechanism
          <select id="filter-mechanism">
            <option value="">Overall grade</option>
            {mechanism_options}
          </select>
        </label>
        {folder_filter_rows}
        <button id="filter-reset" type="button">Reset filters</button>
      </div>
      <p class="sbs-summary" id="sbs-summary">{len(common_keys)} common scenarios across {len(labels)} folder(s)
        ({', '.join(f'{html.escape(l)}: {len(all_tests[l])}' for l in labels)}).</p>
      <table class="sbs-table" id="sbs-table">
        <thead><tr><th>Scenario</th>{header_cols}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
      <script>
        (function() {{
          const table = document.getElementById("sbs-table");
          const summary = document.getElementById("sbs-summary");
          const riskFilter = document.getElementById("filter-risk-category");
          const mechFilter = document.getElementById("filter-mechanism");
          const folderFilters = Array.from(document.querySelectorAll(".folder-grade-filter"));
          const totalRows = table.tBodies[0].rows.length;

          const allCells = Array.from(table.querySelectorAll("td.sbs-cell"));

          function cellGrade(cell, mechanism) {{
            if (!mechanism) return cell.dataset.grade;
            const mechGrades = JSON.parse(cell.dataset.mechGrades || "{{}}");
            return mechGrades[mechanism] || "unknown";
          }}

          function gradeBadgeClass(grade) {{
            return {{
              exemplary: "grade-exemplary",
              adequate: "grade-adequate",
              failing: "grade-failing",
            }}[grade] || "grade-unknown";
          }}

          function updateCellBadges(mechanism) {{
            for (const cell of allCells) {{
              const grade = cellGrade(cell, mechanism);
              const header = cell.querySelector(".sbs-cell-header");
              const label = header.querySelector(".header-label");
              const badge = header.querySelector(".badge");
              label.textContent = mechanism ? mechanism + ": " : "";
              badge.className = "badge " + gradeBadgeClass(grade);
              badge.textContent = grade;
            }}
          }}

          function applyFilters() {{
            const riskValue = riskFilter.value;
            const mechanism = mechFilter.value;
            updateCellBadges(mechanism);
            let visible = 0;
            for (const row of table.tBodies[0].rows) {{
              let show = true;
              if (riskValue && row.dataset.riskCategory !== riskValue) show = false;
              if (show) {{
                for (const sel of folderFilters) {{
                  if (!sel.value) continue;
                  const cell = row.querySelector(`td[data-folder="${{sel.dataset.folder}}"]`);
                  if (!cell || cellGrade(cell, mechanism) !== sel.value) {{ show = false; break; }}
                }}
              }}
              row.style.display = show ? "" : "none";
              if (show) visible++;
            }}
            summary.textContent = `${{visible}} of ${{totalRows}} scenarios match current filters.`;
          }}

          riskFilter.addEventListener("change", applyFilters);
          mechFilter.addEventListener("change", applyFilters);
          folderFilters.forEach(sel => sel.addEventListener("change", applyFilters));
          document.getElementById("filter-reset").addEventListener("click", () => {{
            riskFilter.value = "";
            mechFilter.value = "";
            folderFilters.forEach(sel => sel.value = "");
            applyFilters();
          }});
        }})();
      </script>
    """


def render_bar(count: int, total: int) -> str:
    pct = (count / total * 100) if total else 0
    return (
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
        f'<span class="bar-label">{count} ({pct:.1f}%)</span>'
    )


def render_score_pill(score_pct: float | None, points: int, max_points: int) -> str:
    if score_pct is None:
        return '<span class="score-pill score-pill-empty">n/a</span>'
    return f'<span class="score-pill">{score_pct:.1f}%</span> <span class="stats-total-inline">({points}/{max_points})</span>'


def render_stats_tab(folders: list[Path], all_tests: dict[str, dict[str, dict]]) -> str:
    labels = [folder_label(f) for f in folders]
    all_stats = {label: compute_stats(all_tests[label]) for label in labels}

    header_cols = "".join(
        f"<th>{html.escape(label)} <span class='stats-total-inline'>({all_stats[label]['total']} tests)</span></th>"
        for label in labels
    )

    # Overall grade distribution: one row per grade, one column per folder.
    grade_rows = []
    for g in GRADE_ORDER:
        cells = "".join(
            f"<td>{render_bar(all_stats[label]['grade_counts'].get(g, 0), all_stats[label]['total'])}</td>"
            for label in labels
        )
        grade_rows.append(f"<tr><td>{render_grade_badge(g)}</td>{cells}</tr>")

    # Mechanism grade distribution: one row per mechanism, one column per
    # folder, each cell showing the score (over triggered cases only) plus
    # the full grade breakdown.
    mech_rows = []
    for m in MECHANISMS:
        cells = []
        for label in labels:
            counts = all_stats[label]["mechanism_grade_counts"][m]
            score = all_stats[label]["mechanism_scores"][m]
            grades_cell = " &nbsp;|&nbsp; ".join(
                f"{render_grade_badge(g)} {counts.get(g, 0)}" for g in GRADE_ORDER
            )
            score_cell = render_score_pill(
                score["score_pct"], score["points"], score["max_points"]
            )
            cells.append(
                f"<td>{score_cell}<br>"
                f"<span class='stats-subrow'>{grades_cell}</span></td>"
            )
        mech_rows.append(f"<tr><td>{html.escape(m)}</td>{''.join(cells)}</tr>")

    # Grade by risk category: one row per risk category (union across
    # folders), one column per folder, same score + breakdown layout.
    all_risk_categories = sorted(
        set().union(*(all_stats[label]["risk_category_grades"].keys() for label in labels))
    )
    risk_rows = []
    for risk_category in all_risk_categories:
        cells = []
        for label in labels:
            counts = all_stats[label]["risk_category_grades"].get(risk_category, {})
            risk_total = sum(counts.values())
            if risk_total == 0:
                cells.append('<td class="stats-empty">—</td>')
                continue
            score = all_stats[label]["risk_category_scores"][risk_category]
            grades_cell = " &nbsp;|&nbsp; ".join(
                f"{render_grade_badge(g)} {counts.get(g, 0)}" for g in GRADE_ORDER
            )
            score_cell = render_score_pill(
                score["score_pct"], score["points"], score["max_points"]
            )
            cells.append(
                f"<td>{score_cell}<br>"
                f"<span class='stats-subrow'>{grades_cell} <span class='stats-total-inline'>(n={risk_total})</span></span></td>"
            )
        risk_rows.append(f"<tr><td>{html.escape(risk_category)}</td>{''.join(cells)}</tr>")

    # Total score: points/(2*N)*100, one card per folder.
    score_cards = []
    for label in labels:
        stats = all_stats[label]
        score_cards.append(f"""
          <div class="score-card">
            <div class="score-label">{html.escape(label)}</div>
            <div class="score-value">{stats['overall_score_pct']:.1f}%</div>
            <div class="score-detail">{stats['total_points']} / {stats['max_points']} points across {stats['total']} tests</div>
          </div>
        """)

    return f"""
      <div class="stats-section">
        <h2>Total score</h2>
        <div class="score-cards">{"".join(score_cards)}</div>
      </div>

      <div class="stats-section">
        <h2>Overall grade distribution</h2>
        <table class="stats-table">
          <thead><tr><th></th>{header_cols}</tr></thead>
          <tbody>{"".join(grade_rows)}</tbody>
        </table>
      </div>

      <div class="stats-section">
        <h2>Mechanism scores</h2>
        <p class="stats-note">Score = points / (2 × triggered cases) × 100. "Not triggered" occurrences (precondition not met) are excluded from scoring but shown in the grade breakdown below the score.</p>
        <table class="stats-table">
          <thead><tr><th>Mechanism</th>{header_cols}</tr></thead>
          <tbody>{"".join(mech_rows)}</tbody>
        </table>
      </div>

      <div class="stats-section">
        <h2>Scores by risk category</h2>
        <table class="stats-table">
          <thead><tr><th>Risk category</th>{header_cols}</tr></thead>
          <tbody>{"".join(risk_rows)}</tbody>
        </table>
      </div>
    """


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KORA Results — {title}</title>
<style>
  :root {{
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #666;
    --border: #ddd;
    --card-bg: #f7f7f8;
    --exemplary: #1a7f37;
    --adequate: #b08800;
    --failing: #c02020;
    --unknown: #888;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16181c;
      --fg: #e6e6e6;
      --muted: #999;
      --border: #333;
      --card-bg: #21242a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
  }}
  header {{
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{ margin: 0 0 4px 0; font-size: 1.3em; }}
  header .subtitle {{ color: var(--muted); font-size: 0.9em; }}
  nav.tabs {{
    display: flex;
    gap: 4px;
    padding: 0 24px;
    border-bottom: 1px solid var(--border);
  }}
  nav.tabs button {{
    background: none;
    border: none;
    padding: 10px 18px;
    font-size: 0.95em;
    cursor: pointer;
    color: var(--muted);
    border-bottom: 2px solid transparent;
  }}
  nav.tabs button.active {{
    color: var(--fg);
    border-bottom-color: var(--fg);
    font-weight: 600;
  }}
  main {{ padding: 20px 24px 60px; overflow-x: auto; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid var(--border); padding: 8px 10px; vertical-align: top; text-align: left; }}
  th {{ background: var(--card-bg); }}

  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    font-weight: 600;
    color: white;
  }}
  .grade-exemplary {{ background: var(--exemplary); }}
  .grade-adequate {{ background: var(--adequate); }}
  .grade-failing {{ background: var(--failing); }}
  .grade-unknown {{ background: var(--unknown); }}

  .filters {{
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: 14px;
    padding: 12px 14px;
    margin-bottom: 12px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  .filter-label {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 0.8em;
    color: var(--muted);
  }}
  .filter-label select {{
    font-size: 0.95em;
    padding: 4px 6px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
  }}
  .filters button {{
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
    cursor: pointer;
    font-size: 0.85em;
  }}
  .sbs-summary {{ color: var(--muted); font-size: 0.9em; }}
  .sbs-table {{ min-width: 100%; }}
  .sbs-scenario {{ min-width: 220px; max-width: 280px; }}
  .scenario-title {{ font-weight: 600; margin-bottom: 4px; }}
  .scenario-first-msg {{ color: var(--muted); font-size: 0.85em; margin-bottom: 4px; }}
  .scenario-key {{ color: var(--muted); font-size: 0.7em; font-family: monospace; }}
  .sbs-cell {{ min-width: 480px; max-width: 720px; }}
  .sbs-cell-header {{ margin-bottom: 6px; }}
  .header-label {{ font-size: 0.8em; color: var(--muted); margin-right: 4px; }}
  .cell-body {{ display: flex; gap: 10px; margin-top: 6px; }}
  .cell-body .conversation {{ flex: 1 1 55%; min-width: 0; margin: 0; }}
  .cell-body .assessment-col {{ flex: 1 1 45%; min-width: 0; }}

  .conversation {{ margin: 8px 0; max-height: 420px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; padding: 8px; background: var(--card-bg); }}
  .msg {{ margin-bottom: 8px; padding: 6px 8px; border-radius: 6px; }}
  .msg-user {{ background: rgba(100,150,255,0.12); }}
  .msg-assistant {{ background: rgba(100,255,150,0.10); }}
  .msg-role {{ font-size: 0.7em; text-transform: uppercase; color: var(--muted); font-weight: 700; margin-bottom: 2px; }}
  .msg-content {{ white-space: pre-wrap; font-size: 0.88em; }}

  .assessment {{ font-size: 0.88em; max-height: 420px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; padding: 8px; }}
  .assessment-reasons {{ margin: 6px 0; color: var(--muted); }}
  .mech-table {{ margin-top: 6px; font-size: 0.85em; }}
  .mech-reasons {{ color: var(--muted); }}

  .stats-section {{ margin-bottom: 40px; }}
  .stats-total-inline {{ color: var(--muted); font-weight: normal; font-size: 0.85em; }}
  .stats-table {{ margin-bottom: 20px; }}
  .stats-empty {{ color: var(--muted); text-align: center; }}
  .stats-note {{ color: var(--muted); font-size: 0.85em; margin: -4px 0 12px; }}
  .stats-subrow {{ font-size: 0.9em; }}
  .score-pill {{ font-weight: 700; }}
  .score-pill-empty {{ color: var(--muted); font-weight: normal; }}

  .score-cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .score-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 24px;
    min-width: 180px;
    text-align: center;
  }}
  .score-label {{ font-weight: 600; margin-bottom: 6px; }}
  .score-value {{ font-size: 2.2em; font-weight: 700; }}
  .score-detail {{ color: var(--muted); font-size: 0.8em; margin-top: 4px; }}

  .bar-track {{ display: inline-block; width: 200px; height: 10px; background: var(--border); border-radius: 5px; overflow: hidden; vertical-align: middle; margin-right: 8px; }}
  .bar-fill {{ height: 100%; background: var(--fg); opacity: 0.6; }}
  .bar-label {{ font-size: 0.85em; color: var(--muted); }}

  details summary {{ cursor: pointer; font-size: 0.85em; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>KORA Results Comparison</h1>
  <div class="subtitle">{title}</div>
</header>
<nav class="tabs">
  <button class="tab-btn active" data-tab="sbs">Side-by-side</button>
  <button class="tab-btn" data-tab="stats">Statistics</button>
</nav>
<main>
  <section id="tab-sbs" class="tab-panel active">
    {sbs_html}
  </section>
  <section id="tab-stats" class="tab-panel">
    {stats_html}
  </section>
</main>
<script>
  document.querySelectorAll(".tab-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    }});
  }});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folders", nargs="+", help="Run folders to join and compare")
    parser.add_argument(
        "-o",
        "--output",
        default="data/join_results.html",
        help="Output HTML file path (default: data/join_results.html)",
    )
    parser.add_argument(
        "--safety-cube-jsonl",
        default=None,
        help="Also export a SafetyCube-compatible JSONL (one Conversation record per "
        "test, containing the full multi-turn conversation) to this path — feed "
        "directly to SafetyCube/Safety Grader. No conversion needed on the "
        "SafetyCube side: matches yolo's conversation.Conversation schema.",
    )
    parser.add_argument(
        "--target-model",
        action="append",
        default=[],
        metavar="FOLDER=MODEL",
        help="Record the target model name for a folder in the SafetyCube export "
        "(e.g. --target-model maithinking_results=maithinking) — not recoverable "
        "from the test files themselves, since kora's results.json is often left "
        "empty on interrupted runs. Repeatable, one per folder.",
    )
    parser.add_argument(
        "--user-model",
        action="append",
        default=[],
        metavar="FOLDER=MODEL",
        help="Same as --target-model, for the user-simulator model.",
    )
    args = parser.parse_args()

    def parse_label_model_pairs(pairs: list[str], flag_name: str) -> dict[str, str]:
        result = {}
        for pair in pairs:
            if "=" not in pair:
                raise SystemExit(
                    f"{flag_name} expects FOLDER=MODEL (got: {pair!r})"
                )
            label, model = pair.split("=", 1)
            result[label] = model
        return result

    target_models = parse_label_model_pairs(args.target_model, "--target-model")
    user_models = parse_label_model_pairs(args.user_model, "--user-model")

    folders = [Path(f) for f in args.folders]
    for f in folders:
        if not f.exists():
            raise SystemExit(f"Folder not found: {f}")

    all_tests: dict[str, dict[str, dict]] = {}
    for folder in folders:
        label = folder_label(folder)
        tests = load_folder(folder)
        if label in all_tests:
            raise SystemExit(
                f"Duplicate folder name '{label}' — pass folders with distinct names."
            )
        all_tests[label] = tests
        print(f"Loaded {len(tests)} test(s) from {folder} ({label})")

    if args.safety_cube_jsonl:
        jsonl_path = Path(args.safety_cube_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        n = export_safety_cube_jsonl(
            folders, all_tests, jsonl_path, target_models, user_models
        )
        print(f"Wrote {n} SafetyCube conversation record(s) to {jsonl_path}")

    sbs_html = render_sbs_tab(folders, all_tests)
    stats_html = render_stats_tab(folders, all_tests)

    title = " vs. ".join(folder_label(f) for f in folders)
    page = PAGE_TEMPLATE.format(title=html.escape(title), sbs_html=sbs_html, stats_html=stats_html)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
