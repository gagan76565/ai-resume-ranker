"""Optional end-to-end UI check with Playwright (not needed to run the app).

    pip install playwright && playwright install chromium
    python app.py                      # in another terminal, with ADMIN_* set
    BASE_URL=http://127.0.0.1:5000 OWNER_USER=owner OWNER_PASSWORD=... python scripts/ui_check.py

Checks: no horizontal scroll at 360/768/1280 px, keyboard-only sample flow,
results features (explain, compare, shortlist, sort, CSV, theme), per-file
upload validation, owner history delete + undo, 404 page, and that the
browser console shows no errors (CSP violations would appear there).
Screenshots go to SHOT_DIR (default: a temp folder) for manual review.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from talentlens.minipdf import build_pdf  # noqa: E402

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:5000").rstrip("/")
OWNER = os.environ.get("OWNER_USER")
PASSWORD = os.environ.get("OWNER_PASSWORD")
SHOTS = Path(os.environ.get("SHOT_DIR") or tempfile.mkdtemp(prefix="tl-shots-"))
SHOTS.mkdir(parents=True, exist_ok=True)
WIDTHS = (375, 768, 1024, 1440)
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))


def no_hscroll(page) -> tuple[bool, str]:
    sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    return sw <= iw, f"scrollWidth={sw} innerWidth={iw}"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tl-ui-"))
    good = tmp / "jordan-sample.pdf"
    good.write_bytes(build_pdf(["# Skills", "Python, SQL, Docker, AWS, pandas", "# Experience",
                                "Built REST APIs with Flask and Docker on AWS; wrote SQL for PostgreSQL."]
                               + [f"Delivered analytics project {i} in Python." for i in range(10)]))
    fake = tmp / "not-really.pdf"
    fake.write_bytes(b"this is not a pdf")

    exe = os.environ.get("CHROMIUM_PATH")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page = context.new_page()
        console_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(str(e)))

        # ---- upload page at three widths
        for width in WIDTHS:
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(BASE + "/")
            ok, detail = no_hscroll(page)
            check(f"upload page no horizontal scroll @{width}px", ok, detail)
            page.screenshot(path=str(SHOTS / f"upload-{width}.png"), full_page=True)

        # ---- keyboard-only sample flow
        page.set_viewport_size({"width": 1280, "height": 900})
        page.goto(BASE + "/")
        seen = []
        for _ in range(40):
            page.keyboard.press("Tab")
            focused = page.evaluate("document.activeElement && (document.activeElement.id || document.activeElement.textContent.trim().slice(0,30))")
            seen.append(focused)
            if focused == "try-samples":
                break
        check("keyboard reaches 'Try with sample resumes'", "try-samples" in seen, f"{len(seen)} tabs")
        outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
        check("visible focus outline on focused button", outline not in ("none", ""), outline)
        page.keyboard.press("Enter")
        expect(page.locator("#sample-notice")).to_be_visible()
        check("samples loaded via keyboard", page.locator("#file-list li").count() == 6)
        check("submit button focused after loading samples", page.evaluate("document.activeElement.id") == "submit-btn")
        page.keyboard.press("Enter")
        page.wait_for_url("**/results")
        page.wait_for_selector("#results-body tr")
        rows = page.locator("#results-body > tr:not(.detail-row)").count()
        check("results show 6 ranked sample candidates", rows == 6, f"rows={rows}")
        check("synthetic data banner visible", page.locator("#sample-banner").is_visible())
        check("score shown as number and text label", page.locator(".score-value").first.inner_text().replace(".", "").isdigit()
              and page.locator(".score-label").first.inner_text() != "")

        for width in WIDTHS:
            page.set_viewport_size({"width": width, "height": 900})
            ok, detail = no_hscroll(page)
            check(f"results page no horizontal scroll @{width}px", ok, detail)
            page.screenshot(path=str(SHOTS / f"results-{width}.png"), full_page=True)
        page.set_viewport_size({"width": 1280, "height": 900})

        # explain panel
        page.get_by_role("button", name="Why this score?").first.click()
        marks = page.locator(".detail-row:not([hidden]) mark").count()
        check("'Why this score?' panel shows highlighted terms", marks > 0, f"marks={marks}")
        check("explain button exposes aria-expanded", page.locator('[aria-expanded="true"]').count() == 1)
        page.screenshot(path=str(SHOTS / "results-explain-1280.png"), full_page=True)

        # sorting
        page.locator('th[data-sort="name"] .sort-btn').click()
        check("sort by name sets aria-sort", page.locator('th[data-sort="name"]').get_attribute("aria-sort") == "ascending")
        page.locator('th[data-sort="score"] .sort-btn').click()

        # shortlist
        star = page.locator(".star-btn").first
        star.click()
        check("shortlist toggle pressed", page.locator('.star-btn[aria-pressed="true"]').count() == 1)
        page.locator("#f-shortlist").check()
        shown = page.locator("#results-body > tr:not(.detail-row)").count()
        check("shortlisted-only filter", shown == 1, page.locator("#result-count").inner_text())
        page.locator("#f-shortlist").uncheck()

        # filter empty state
        page.fill("#f-search", "zzzz-nobody")
        check("filter empty state shown", page.locator("#filter-empty").is_visible())
        page.click("#clear-filters")

        # compare
        boxes = page.locator('.compare-check input')
        boxes.nth(0).check()
        boxes.nth(1).check()
        check("third compare box disabled when two selected", boxes.nth(2).is_disabled())
        page.click("#compare-btn")
        expect(page.locator("#compare-dialog")).to_be_visible()
        check("compare dialog shows two columns", page.locator(".compare-col").count() == 2)
        page.screenshot(path=str(SHOTS / "compare-1280.png"))
        page.keyboard.press("Escape")
        check("compare dialog closes with Escape", not page.locator("#compare-dialog").is_visible())

        # CSV export
        with page.expect_download() as info:
            page.click("#export-btn")
        csv_text = Path(info.value.path()).read_text(encoding="utf-8-sig")
        check("CSV export header matches v1.x columns first", csv_text.startswith("Candidate PDF,Text similarity (%),Matched terms"))
        check("CSV export has 6 data rows", len(csv_text.strip().splitlines()) == 7)

        # theme toggle
        page.click("#theme-toggle")
        check("dark theme toggle", page.evaluate("document.documentElement.dataset.theme") == "dark")
        page.screenshot(path=str(SHOTS / "results-dark-1280.png"), full_page=True)
        page.reload()
        check("theme persists across reload", page.evaluate("document.documentElement.dataset.theme") == "dark")
        page.click("#theme-toggle")

        # ---- per-file validation on the upload page
        page.goto(BASE + "/")
        page.set_input_files("#file-input", [str(good), str(fake)])
        page.wait_for_selector(".file-chip.is-invalid")
        check("invalid file flagged inline by name", "not-really.pdf" in page.locator(".file-chip.is-invalid").inner_text())
        page.locator(".file-chip.is-invalid button").click()
        check("remove file chip", page.locator(".file-chip").count() == 1)
        check("rank button disabled until a job description is entered", page.locator("#submit-btn").is_disabled())
        page.fill("#job-description", "too short")
        page.keyboard.press("Tab")
        check("short job description error shown", page.locator("#jd-error").is_visible())
        page.fill("#job-description", "Data engineer with Python, SQL and Docker experience.")
        check("rank button enabled once the form is ready", page.locator("#submit-btn").is_enabled())
        check("workflow step 3 becomes current", page.locator("#step-rank").get_attribute("aria-current") == "step")

        # ---- 404 (the browser logs the expected 404; exclude it from the console check)
        before_404 = len(console_errors)
        resp = page.goto(BASE + "/no-such-page")
        check("404 page", resp.status == 404 and "could not find" in page.content())
        del console_errors[before_404:]

        # ---- owner flow
        if OWNER and PASSWORD:
            page.goto(BASE + "/login")
            page.fill("#username", OWNER)
            page.fill("#password", PASSWORD)
            page.click("button[type=submit]")
            page.wait_for_url(BASE + "/")
            check("owner login", page.locator("text=Sign out").count() == 1)
            page.fill("#job-title", "UI check role")
            page.fill("#job-description", "Data engineer with Python, SQL, Docker and AWS experience building REST APIs.")
            page.set_input_files("#file-input", [str(good)])
            page.wait_for_selector(".file-chip:not(.is-invalid) >> text=Ready")
            page.click("#submit-btn")
            page.wait_for_url("**/screenings/*")
            page.wait_for_selector("#results-body tr")
            check("owner screening saved and opened", "/screenings/" in page.url)
            page.get_by_role("button", name="Why this score?").first.click()
            page.fill(".notes textarea", "Strong SQL")
            page.click("text=Save note")
            page.wait_for_selector("text=Note saved.")
            check("owner note saved", True)
            page.goto(BASE + "/history")
            page.wait_for_selector(".history-item")
            before = page.locator(".history-item").count()
            for width in WIDTHS:
                page.set_viewport_size({"width": width, "height": 900})
                ok, detail = no_hscroll(page)
                check(f"history page no horizontal scroll @{width}px", ok, detail)
                page.screenshot(path=str(SHOTS / f"history-{width}.png"), full_page=True)
            page.set_viewport_size({"width": 1280, "height": 900})
            page.locator(".history-item button", has_text="Delete").first.click()
            check("delete hides item immediately", page.locator(".history-item").count() == before - 1)
            page.get_by_role("button", name="Undo").click()
            check("undo restores item", page.locator(".history-item").count() == before)
            page.fill("#h-search", "UI check role")
            page.wait_for_timeout(600)
            check("history search", page.locator(".history-item").count() >= 1)
        else:
            print("SKIP owner flow (set OWNER_USER and OWNER_PASSWORD)")

        # reduced motion
        context2 = browser.new_context(reduced_motion="reduce")
        p2 = context2.new_page()
        p2.goto(BASE + "/results")
        duration = p2.evaluate("getComputedStyle(document.querySelector('.skeleton')).animationDuration")
        check("prefers-reduced-motion shortens animations", duration in ("1e-06s", "0.001ms", "1e-6s") or duration.startswith("0"), duration)

        check("no console errors or CSP violations", not console_errors, "; ".join(console_errors[:3]))
        browser.close()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed. Screenshots: {SHOTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
