# Screenshots to capture

The README links to these files. None exist yet: please capture them from a
real run and save them as PNG in `docs/screenshots/` with exactly these names.
Use the **synthetic sample data** only, never real resumes.

Suggested setup: `python app.py`, a browser window 1280 px wide (or 390 px for
mobile shots), and the zoom at 100%.

| File | Page / state | How to get there |
| --- | --- | --- |
| `docs/screenshots/upload-light.png` | Upload screen, light theme | Open `/`, press **Try with sample resumes** so the six sample chips and must-have tags show |
| `docs/screenshots/results-light.png` | Results table, light theme | Press **Rank candidates** after loading the samples |
| `docs/screenshots/results-why.png` | "Why this score?" panel open | On the results, open **Why this score?** for rank #1 (shows the formula, breakdown and `<mark>` highlights) |
| `docs/screenshots/compare.png` | Compare dialog | Tick **Compare** on rows #1 and #2, then press **Compare (2/2)** |
| `docs/screenshots/results-dark.png` | Results, dark theme | Use the moon toggle in the header |
| `docs/screenshots/mobile-results.png` | Results at mobile width | 390 px wide viewport (DevTools device mode) |
| `docs/screenshots/scanned-warning.png` | Low-text warning | Open **Why this score?** for `sample-06-scanned-simulation.pdf` |
| `docs/screenshots/history.png` | Owner history | Sign in as the owner, run one screening with your own **synthetic** PDFs, then open **History** |
| `docs/screenshots/login.png` | Owner sign in | `/login` |

Tip: `python scripts/ui_check.py` (needs Playwright) saves reference screenshots to a
temporary folder. They are for checking the layout and are not meant for the README.
