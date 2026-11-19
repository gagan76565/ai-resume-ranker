/* Results workspace: summary, skill gaps, requirements, filters, sorting,
   explanations, shortlist, compare, CSV, print. All values shown come from
   the API response; nothing is invented client-side. */
(function () {
  "use strict";
  var TL = window.TL, $ = TL.$, el = TL.el;

  var root = $("#results-root");
  var screeningId = root.dataset.screeningId ? Number(root.dataset.screeningId) : null;
  var isOwner = document.body.dataset.owner === "true";
  var data = null;
  var view = { q: "", minScore: 0, minCov: 0, shortOnly: false, sort: "score", dir: "desc" };
  var expanded = {};
  var compare = [];
  var firstPaint = true;
  var BANDS = [
    { key: "strong", label: "Strong 70+", min: 70 },
    { key: "partial", label: "Partial 45–69", min: 45 },
    { key: "weak", label: "Weak 20–44", min: 20 },
    { key: "very-weak", label: "Very weak <20", min: -1 }
  ];

  function keyOf(c) { return c.id ? "id-" + c.id : "f-" + c.filename; }
  function nameOf(c) { return c.display_name || c.filename.replace(/\.pdf$/i, "").replace(/[_-]+/g, " "); }
  function cssId(key) { return key.replace(/[^a-zA-Z0-9_-]/g, "_"); }
  function pct(n) { return n === null || n === undefined ? "—" : Number(n).toFixed(1); }
  function weightsOf(b) { return b.weights || { tfidf: 1, skills: 0, sections: 0 }; }
  function isLegacy(b) { return b.skill_coverage === null || b.skill_coverage === undefined; }
  function bandOf(score) { return TL.band(score); }
  function bar(value, cls, animate) {
    var fill = el("span", { class: cls || "" });
    TL.setWidth(fill, value, animate);
    return el("div", { class: "meter", "aria-hidden": "true" }, [fill]);
  }

  /* ---------------------------------------------------------------- load */
  function loadLocal() {
    var stored = TL.store.get("sessionStorage", "tl:last-result");
    if (!stored && typeof window.name === "string" && window.name.indexOf("tl:") === 0) {
      try { stored = JSON.parse(window.name.slice(3)); } catch (e) { stored = null; }
    }
    return stored;
  }
  function persistLocal() {
    if (screeningId) return;
    if (!TL.store.set("sessionStorage", "tl:last-result", data)) {
      try { window.name = "tl:" + JSON.stringify(data); } catch (e) { /* ignore */ }
    }
  }
  function showState(which) {
    $("#results-loading").hidden = which !== "loading";
    $("#results-empty").hidden = which !== "empty";
    $("#results-content").hidden = which !== "content";
  }
  function start() {
    if (screeningId) {
      TL.api("/api/v1/screenings/" + screeningId).then(function (d) { data = d; ready(); }).catch(function (err) {
        showState("empty");
        $("#empty-title").textContent = err.status === 404 ? "Screening not found" : "Could not load this screening";
        $("#empty-text").textContent = err.status === 404 ? "It may have been deleted or removed by the retention policy." : err.message;
        $("#results-status").textContent = "Results could not be loaded.";
      });
    } else {
      data = loadLocal();
      if (!data || !data.candidates) { showState("empty"); $("#results-status").textContent = "No results in this tab."; return; }
      ready();
    }
  }

  function ready() {
    showState("content");
    $("#results-title").textContent = data.job_title || "Untitled role";
    document.title = (data.job_title || "Results") + " · TalentLens";
    renderMeta();
    $("#sample-banner").hidden = !data.sample_data;
    renderFileErrors();
    renderSummary();
    renderGaps();
    renderRequirements();
    render();
    firstPaint = false;
    $("#results-status").textContent = "Loaded " + data.candidates.length + " candidates.";
  }

  /* ---------------------------------------------------------------- header + summary */
  function metaChip(iconName, text, cls) { return el("span", { class: "meta-chip" + (cls ? " " + cls : "") }, [TL.icon(iconName), text]); }
  function renderMeta() {
    var meta = $("#results-meta");
    meta.textContent = "";
    meta.appendChild(metaChip("calendar", (data.saved ? "Saved " : "Screened ") + TL.formatDate(data.created_at)));
    meta.appendChild(metaChip("cpu", "Scoring v" + (data.scoring_version || "2.0")));
    meta.appendChild(data.saved ? metaChip("lock", "Saved to owner history", "is-accent") : metaChip("shield", "Not stored on the server"));
    if (data.sample_data) meta.appendChild(metaChip("info", "Synthetic demo data"));
  }

  function renderFileErrors() {
    var box = $("#file-errors");
    var errs = data.file_errors || [];
    box.textContent = "";
    if (!errs.length) { box.hidden = true; return; }
    box.appendChild(TL.icon("alert"));
    box.appendChild(el("div", {}, [
      el("strong", { text: errs.length + " file" + (errs.length === 1 ? " was" : "s were") + " skipped" }),
      el("ul", {}, errs.map(function (e) { return el("li", { text: e.error }); }))
    ]));
    box.hidden = false;
  }

  function renderSummary() {
    var cs = data.candidates;
    var n = cs.length;
    var avg = n ? cs.reduce(function (s, c) { return s + c.score; }, 0) / n : 0;
    var top = cs.slice().sort(function (a, b) { return b.score - a.score || a.rank - b.rank; })[0];
    $("#sum-count").textContent = String(n);
    var dist = $("#sum-dist"), legend = $("#sum-legend");
    dist.textContent = ""; legend.textContent = "";
    var parts = [];
    BANDS.forEach(function (band) {
      var count = cs.filter(function (c) { return bandOf(c.score) === band.key; }).length;
      if (count) {
        var seg = el("span", { class: "band-" + band.key });
        seg.style.flexGrow = String(count);
        dist.appendChild(seg);
      }
      legend.appendChild(el("li", {}, [el("i", { class: "band-" + band.key }), band.label + ": " + count]));
      parts.push(band.label + " " + count);
    });
    dist.setAttribute("aria-label", "Score bands: " + parts.join(", "));
    $("#sum-average").textContent = avg.toFixed(1);
    TL.setWidth($("#sum-avg-bar"), avg, true);
    $("#sum-avg-foot").textContent = n ? "Band: " + (avg >= 70 ? "strong" : avg >= 45 ? "partial" : avg >= 20 ? "weak" : "very weak") + " match on average" : "";
    $("#sum-top").textContent = top ? top.score.toFixed(1) : "0";
    TL.setWidth($("#sum-top-bar"), top ? top.score : 0, true);
    var foot = $("#sum-top-foot");
    foot.textContent = "";
    if (top) { foot.appendChild(document.createTextNode("Ranked #1: ")); foot.appendChild(el("b", { text: nameOf(top) })); }
  }

  function renderGaps() {
    var list = $("#sum-missing");
    list.textContent = "";
    var n = data.candidates.length;
    var missing = (data.summary && data.summary.top_missing_skills) || [];
    if (!missing.length) {
      list.appendChild(el("li", { class: "muted small", text: "No required skills are missing across this pool." }));
      return;
    }
    $("#gaps-sub").textContent = "Required skills most often missing across " + n + " candidate" + (n === 1 ? "" : "s") + ".";
    missing.forEach(function (m) {
      list.appendChild(el("li", { class: "gap" }, [
        el("div", { class: "gap-row" }, [el("strong", { text: m.skill }), el("span", { text: "missing in " + m.missing_count + " of " + n })]),
        bar(n ? (m.missing_count / n) * 100 : 0, "", true)
      ]));
    });
  }

  function renderRequirements() {
    var w = data.weights || (data.candidates[0] && data.candidates[0].breakdown.weights) || null;
    var req = data.required_skills || [];
    var must = data.must_have || [];
    $("#required-count").textContent = String(req.length);
    var chips = $("#summary-chips");
    chips.textContent = "";
    chips.appendChild(el("span", { class: "badge badge-neutral", text: req.length + " required skills" }));
    if (must.length) chips.appendChild(el("span", { class: "badge badge-accent", text: must.length + " must-have" + (must.length === 1 ? "" : "s") }));
    $("#weights-text").textContent = w ? " · Text " + Math.round(w.tfidf * 100) + "% · Skills " + Math.round((w.skills || 0) * 100) + "% · Sections " + Math.round((w.sections || 0) * 100) + "%"
      : " · Imported from v1.x: text similarity only";

    var list = $("#required-skills");
    list.textContent = "";
    if (!req.length) list.appendChild(el("li", { class: "muted small", text: "No known skills were detected, so scores use text similarity only." }));
    req.forEach(function (s) {
      var isMust = must.indexOf(s) !== -1;
      list.appendChild(el("li", { class: "chip " + (isMust ? "chip-accent" : "chip-neutral") }, [isMust ? TL.icon("star-fill") : null, s + (isMust ? " · must-have" : "")]));
    });

    var wb = $("#weight-bar"), wl = $("#weight-list");
    wb.textContent = ""; wl.textContent = "";
    if (w) {
      [["tfidf", "Text similarity (TF-IDF)"], ["skills", "Skill coverage"], ["sections", "Section signal"]].forEach(function (pair) {
        var value = w[pair[0]] || 0;
        if (value > 0) { var seg = el("span", { class: "w-" + pair[0] }); seg.style.flexGrow = String(value); wb.appendChild(seg); }
        wl.appendChild(el("li", {}, [el("i", { class: "w-" + pair[0] }), el("span", { text: pair[1] }), el("b", { text: Math.round(value * 100) + "%" })]));
      });
    }
    var cap = data.settings && data.settings.must_have_cap;
    $("#cap-text").textContent = must.length ? "A candidate missing any must-have skill has their score capped" + (cap ? " at " + cap : "") + "." : "No must-have skills were set for this screening.";
    $("#job-description-text").textContent = data.job_description || "";
  }
  TL.tabs($("#req-tabs"));

  /* ---------------------------------------------------------------- filtering + sorting */
  function visibleCandidates() {
    var q = view.q.toLowerCase();
    var rows = data.candidates.filter(function (c) {
      var b = c.breakdown;
      if (q && (nameOf(c) + " " + c.filename).toLowerCase().indexOf(q) === -1) return false;
      if (c.score < view.minScore) return false;
      if (view.minCov && (b.skill_coverage || 0) < view.minCov) return false;
      if (view.shortOnly && !c.shortlisted) return false;
      return true;
    });
    var dir = view.dir === "asc" ? 1 : -1;
    rows.sort(function (a, b) {
      var x, y;
      if (view.sort === "name") { x = nameOf(a).toLowerCase(); y = nameOf(b).toLowerCase(); return x < y ? -dir : x > y ? dir : 0; }
      if (view.sort === "rank") { x = a.rank; y = b.rank; }
      else if (view.sort === "coverage") { x = a.breakdown.skill_coverage || 0; y = b.breakdown.skill_coverage || 0; }
      else { x = a.score; y = b.score; }
      return x === y ? a.rank - b.rank : (x - y) * dir;
    });
    return rows;
  }
  function syncSortHeaders() {
    TL.$$("#results-table th[data-sort]").forEach(function (th) {
      var active = th.dataset.sort === view.sort;
      th.setAttribute("aria-sort", active ? (view.dir === "asc" ? "ascending" : "descending") : "none");
      th.querySelector(".arrow").textContent = active ? (view.dir === "asc" ? "↑" : "↓") : "↕";
    });
  }
  TL.$$("#results-table th[data-sort] .sort-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var key = btn.parentNode.dataset.sort;
      if (view.sort === key) view.dir = view.dir === "asc" ? "desc" : "asc";
      else { view.sort = key; view.dir = key === "name" || key === "rank" ? "asc" : "desc"; }
      render();
    });
  });
  $("#f-search").addEventListener("input", function (e) { view.q = e.target.value.trim(); render(); });
  $("#f-score").addEventListener("change", function (e) { view.minScore = Number(e.target.value); render(); });
  $("#f-coverage").addEventListener("change", function (e) { view.minCov = Number(e.target.value); render(); });
  $("#f-shortlist").addEventListener("change", function (e) { view.shortOnly = e.target.checked; render(); });
  $("#clear-filters").addEventListener("click", function () {
    view.q = ""; view.minScore = 0; view.minCov = 0; view.shortOnly = false;
    $("#f-search").value = ""; $("#f-score").value = "0"; $("#f-coverage").value = "0"; $("#f-shortlist").checked = false;
    render(); $("#f-search").focus();
  });

  /* ---------------------------------------------------------------- table */
  function scoreCell(c, animate) {
    var b = c.breakdown;
    var fill = el("span");
    TL.setWidth(fill, c.score, animate);
    var track = el("span", { class: "score-bar", "aria-hidden": "true" }, [fill]);
    [20, 45, 70].forEach(function (t) { var tick = el("i"); tick.style.left = t + "%"; track.appendChild(tick); });
    return el("div", { class: "score", dataset: { band: bandOf(c.score) } }, [
      el("span", { class: "score-value", text: c.score.toFixed(1) }),
      track,
      el("span", { class: "score-label", text: (b.label || "") + (b.capped ? " · capped" : "") })
    ]);
  }
  function coverageCell(c, animate) {
    var b = c.breakdown;
    if (isLegacy(b)) return el("div", { class: "cov" }, [el("span", { class: "cov-text muted", text: "n/a (v1 import)" })]);
    var total = (b.required_skills || []).length;
    if (!total) return el("div", { class: "cov" }, [el("span", { class: "cov-text muted", text: "n/a" })]);
    return el("div", { class: "cov" }, [
      el("span", { class: "cov-text" }, [(b.matched_skills || []).length + "/" + total, el("small", { text: Math.round(b.skill_coverage) + "%" })]),
      bar(b.skill_coverage, "", animate)
    ]);
  }
  function chipList(items, cls, limit, iconName) {
    var shown = items.slice(0, limit);
    var ul = el("ul", { class: "chips" }, shown.map(function (s) { return el("li", { class: "chip " + cls }, [iconName ? TL.icon(iconName) : null, s]); }));
    if (items.length > limit) ul.appendChild(el("li", { class: "chip chip-more", text: "+" + (items.length - limit) + " more" }));
    return ul;
  }
  function flags(c) {
    var b = c.breakdown, out = [];
    if (c.shortlisted) out.push(el("span", { class: "badge badge-accent" }, [TL.icon("star-fill"), "Shortlisted"]));
    if (b.low_text) out.push(el("span", { class: "badge badge-warn" }, [TL.icon("alert"), "Likely scanned"]));
    if (b.capped || (b.must_have_missing && b.must_have_missing.length)) out.push(el("span", { class: "badge badge-danger", text: "Missing must-have" }));
    return out.length ? el("div", { class: "cand-flags" }, out) : null;
  }
  function starButton(c) {
    var label = (c.shortlisted ? "Remove " : "Add ") + nameOf(c) + (c.shortlisted ? " from shortlist" : " to shortlist");
    return el("button", {
      type: "button", class: "btn btn-ghost btn-icon btn-sm star-btn", "aria-pressed": c.shortlisted ? "true" : "false",
      "aria-label": label, "data-tip": c.shortlisted ? "Shortlisted" : "Shortlist", onclick: function () { toggleShortlist(c); }
    }, [TL.icon(c.shortlisted ? "star-fill" : "star")]);
  }
  function compareBox(c) {
    var key = keyOf(c);
    var checked = compare.indexOf(key) !== -1;
    var input = el("input", { type: "checkbox", id: "cmp-" + cssId(key), checked: checked, disabled: !checked && compare.length >= 2,
      "aria-label": "Select " + nameOf(c) + " for comparison", onchange: function (e) {
        if (e.target.checked) { if (compare.length < 2) compare.push(key); }
        else compare = compare.filter(function (k) { return k !== key; });
        render();
        var again = document.getElementById("cmp-" + cssId(key));
        if (again) again.focus();
      } });
    return el("label", { class: "compare-check" }, [input, el("span", { class: "cmp-text", text: "Compare" })]);
  }

  function render() {
    syncSortHeaders();
    var rows = visibleCandidates();
    var body = $("#results-body");
    body.textContent = "";
    rows.forEach(function (c) {
      var b = c.breakdown;
      var key = keyOf(c);
      var detailId = "detail-" + cssId(key);
      var isOpen = !!expanded[key];
      var whyBtn = el("button", { type: "button", class: "btn btn-secondary btn-sm details-btn", "aria-expanded": isOpen ? "true" : "false",
        "aria-controls": detailId, onclick: function () {
          expanded[key] = !expanded[key]; render();
          var again = document.querySelector('[aria-controls="' + detailId + '"]');
          if (again) again.focus();
        } }, [el("span", { text: isOpen ? "Hide details" : "Why this score?" }), TL.icon("chevron")]);
      var matched = b.matched_skills || [], missing = b.missing_skills || [];
      body.appendChild(el("tr", { class: "cand-row" + (c.shortlisted ? " is-shortlisted" : "") + (isOpen ? " is-open" : "") }, [
        el("td", { class: "td-rank", "data-label": "Rank" }, [el("span", { class: "rank" + (c.rank === 1 ? " is-top" : ""), "aria-label": "Rank " + c.rank, text: String(c.rank) })]),
        el("td", { class: "td-cand", "data-label": "Candidate" }, [el("div", { class: "cand" }, [
          el("span", { class: "avatar", "aria-hidden": "true", text: TL.initials(nameOf(c)) }),
          el("div", { class: "cand-text" }, [el("div", { class: "cand-name", text: nameOf(c), title: nameOf(c) }), el("div", { class: "cand-file", text: c.filename, title: c.filename }), flags(c)])
        ])]),
        el("td", { class: "td-score", "data-label": "Fit score" }, [scoreCell(c, firstPaint)]),
        el("td", { class: "td-cov", "data-label": "Coverage" }, [coverageCell(c, firstPaint)]),
        el("td", { class: "td-skills", "data-label": "Skills" }, [el("div", { class: "skills-cell" }, [
          matched.length ? chipList(matched, "chip-ok", 3, "check") : el("span", { class: "muted xs", text: "No required skills matched" }),
          missing.length ? el("div", { class: "missing-line", title: "Missing: " + missing.join(", ") }, [el("b", { text: "Missing " + missing.length + ": " }), missing.slice(0, 4).join(", ") + (missing.length > 4 ? "…" : "")]) : el("div", { class: "missing-line", text: "No required skills missing" })
        ])]),
        el("td", { class: "td-act", "data-label": "Actions" }, [el("div", { class: "row-actions" }, [starButton(c), compareBox(c), whyBtn])])
      ]));
      body.appendChild(el("tr", { class: "detail-row", id: detailId, hidden: !isOpen }, [el("td", { colspan: "6" }, [isOpen ? detailPanel(c) : null])]));
    });
    var total = data.candidates.length;
    $("#result-count").textContent = "Showing " + rows.length + " of " + total + " candidate" + (total === 1 ? "" : "s");
    $("#filter-empty").hidden = rows.length > 0;
    $("#results-table").hidden = rows.length === 0;
    var cmp = $("#compare-btn");
    $("#compare-count").textContent = compare.length + "/2";
    $("#compare-count").classList.toggle("is-full", compare.length === 2);
    cmp.disabled = compare.length !== 2;
    $("#compare-hint").textContent = compare.length === 2 ? "Two candidates selected. Press Compare." :
      compare.length === 1 ? "Select one more candidate to compare." : "Select two candidates to compare them side by side.";
  }

  /* ---------------------------------------------------------------- detail panel */
  function panel(title, iconName, content, extra) {
    return el("section", { class: "panel" }, [
      el("div", { class: "panel-head" }, [el("h3", {}, [TL.icon(iconName), title]), extra || null]),
      el("div", {}, content)
    ]);
  }
  function component(name, cls, value, weight, points) {
    return el("div", { class: "component" }, [
      el("div", { class: "component-top" }, [el("span", { text: name }), el("b", { text: pct(value) })]),
      bar(value, cls, true),
      el("div", { class: "component-foot", text: "Weight " + Math.round((weight || 0) * 100) + "% · contributes " + pct(points) + " points" })
    ]);
  }
  function snippetsOf(b, limit) {
    return (b.snippets || []).slice(0, limit || 10).map(function (s) {
      var p = el("p", { class: "snippet" }, [el("span", { class: "snippet-label", text: s.skill })]);
      if (s.prefix_ellipsis) p.appendChild(document.createTextNode("…"));
      s.segments.forEach(function (seg) { p.appendChild(seg.mark ? el("mark", { text: seg.text }) : document.createTextNode(seg.text)); });
      if (s.suffix_ellipsis) p.appendChild(document.createTextNode("…"));
      return p;
    });
  }
  function detailPanel(c) {
    var b = c.breakdown, w = weightsOf(b), contrib = b.contributions || {};
    var legacy = isLegacy(b);
    var ring = el("div", { class: "ring", dataset: { band: bandOf(c.score) } }, [el("b", { text: c.score.toFixed(1) })]);
    ring.style.setProperty("--p", String(Math.max(0, Math.min(100, c.score))));
    var overview = panel("Score overview", "gauge", [
      el("div", { class: "overview" }, [ring, el("div", {}, [
        el("strong", { text: b.label || "" }),
        el("p", { text: legacy ? "Imported from v1.x: TF-IDF similarity only." : (b.capped ? "Capped because a must-have skill is missing." : "Weighted blend of three signals.") })
      ])]),
      legacy ? null : el("p", { class: "formula", text: c.score.toFixed(1) + " = " + Math.round(w.tfidf * 100) + "% × " + pct(b.tfidf_score) + " + " +
        Math.round(w.skills * 100) + "% × " + pct(b.skill_coverage) + " + " + Math.round(w.sections * 100) + "% × " + pct(b.section_signal) + (b.capped ? "  → capped" : "") })
    ]);
    var breakdown = legacy ? null : panel("Score breakdown", "sliders", [el("div", { class: "components" }, [
      component("Text similarity (TF-IDF)", "w-tfidf", b.tfidf_score, w.tfidf, contrib.tfidf),
      component("Skill coverage", "w-skills", b.skill_coverage, w.skills, contrib.skills),
      component("Section signal", "w-sections", b.section_signal, w.sections, contrib.sections)
    ])]);
    var facts = panel("Extracted text", "file", [el("dl", { class: "facts" }, [
      el("div", {}, [el("dt", { text: "Characters" }), el("dd", { text: (b.text_chars || 0).toLocaleString() })]),
      el("div", {}, [el("dt", { text: "Skills in sections" }), el("dd", { text: (b.skills_in_sections || []).length + " of " + (b.matched_skills || []).length })]),
      el("div", {}, [el("dt", { text: "Rank" }), el("dd", { text: "#" + c.rank })])
    ]), (b.warnings || []).length ? el("ul", { class: "warnings mt-3" }, b.warnings.map(function (x) { return el("li", {}, [TL.icon("alert"), el("span", { text: x })]); })) : null]);

    var matched = b.matched_skills || [], missing = b.missing_skills || [], must = b.must_have_missing || [];
    var skills = panel("Skills", "target", [el("div", { class: "skill-cols" }, [
      el("div", {}, [el("div", { class: "section-title mb-2", text: "Matched · " + matched.length }), matched.length ? chipList(matched, "chip-ok", 100, "check") : el("p", { class: "muted xs", text: "None." })]),
      el("div", {}, [el("div", { class: "section-title mb-2", text: "Missing · " + missing.length }), missing.length ? el("ul", { class: "chips" }, missing.map(function (s) {
        var isMust = must.indexOf(s) !== -1;
        return el("li", { class: "chip " + (isMust ? "chip-accent" : "chip-missing") }, [isMust ? TL.icon("star-fill") : TL.icon("x"), s + (isMust ? " · must-have" : "")]);
      })) : el("p", { class: "muted xs", text: "None." })])
    ])]);
    var ev = snippetsOf(b);
    var evidence = panel("Evidence", "quote", ev.length ? ev : [el("p", { class: "muted xs", text: "No excerpts available." })],
      el("span", { class: "count", text: ev.length + " excerpt" + (ev.length === 1 ? "" : "s") }));
    var terms = panel("Top shared terms", "search", [(b.top_terms || []).length ? chipList(b.top_terms, "chip-neutral", 12) : el("p", { class: "muted xs", text: "No shared terms." })]);
    return el("div", { class: "detail" }, [
      el("div", { class: "detail-col" }, [overview, breakdown, facts]),
      el("div", { class: "detail-col" }, [skills, evidence, terms, notesBox(c)])
    ]);
  }
  function notesBox(c) {
    if (!(screeningId && isOwner && c.id)) return null;
    var id = "notes-" + c.id;
    var area = el("textarea", { class: "textarea", id: id, maxlength: "5000", placeholder: "Only visible to you" });
    area.value = c.notes || "";
    var save = el("button", { type: "button", class: "btn btn-secondary btn-sm mt-2", text: "Save note", onclick: function () {
      save.disabled = true;
      TL.api("/api/v1/screenings/" + screeningId + "/candidates/" + c.id, { method: "PATCH", json: { notes: area.value } })
        .then(function (updated) { c.notes = updated.notes; TL.toast("Note saved."); })
        .catch(function (err) { TL.toast(err.message, { type: "error" }); })
        .then(function () { save.disabled = false; });
    } });
    return el("section", { class: "panel notes" }, [el("div", { class: "panel-head" }, [el("h3", {}, [TL.icon("lock"), el("label", { for: id, text: "Private notes" })])]), area, save]);
  }

  /* ---------------------------------------------------------------- shortlist */
  function toggleShortlist(c) {
    var next = !c.shortlisted;
    c.shortlisted = next;
    render();
    var focusStar = function () { var s = TL.$$(".star-btn").filter(function (x) { return x.getAttribute("aria-label").indexOf(nameOf(c)) !== -1; })[0]; if (s) s.focus(); };
    focusStar();
    if (screeningId && c.id) {
      TL.api("/api/v1/screenings/" + screeningId + "/candidates/" + c.id, { method: "PATCH", json: { shortlisted: next } })
        .then(function () { TL.toast(next ? nameOf(c) + " added to the shortlist." : nameOf(c) + " removed from the shortlist."); })
        .catch(function (err) { c.shortlisted = !next; render(); focusStar(); TL.toast(err.message, { type: "error" }); });
    } else {
      persistLocal();
      TL.toast(next ? nameOf(c) + " shortlisted (this tab only)." : nameOf(c) + " removed from the shortlist.");
    }
  }

  /* ---------------------------------------------------------------- compare */
  function pair() {
    return compare.map(function (k) { return data.candidates.filter(function (c) { return keyOf(c) === k; })[0]; }).filter(Boolean);
  }
  function cmpValue(value, other, cls, suffix) {
    var better = value !== null && other !== null && value > other + 0.05;
    return el("div", { class: "cmp-val" }, [
      el("div", { class: "row" }, [el("b", { text: pct(value) + (suffix || "") }), better ? el("span", { class: "delta", text: "+" + (value - other).toFixed(1) }) : null]),
      value === null || value === undefined ? null : bar(value, cls, false)
    ]);
  }
  function openCompare() {
    var p = pair();
    if (p.length !== 2) return;
    var A = p[0], B = p[1], a = A.breakdown, b = B.breakdown;
    var shared = (a.matched_skills || []).filter(function (s) { return (b.matched_skills || []).indexOf(s) !== -1; });
    function only(x, y) { return (x.matched_skills || []).filter(function (s) { return (y.matched_skills || []).indexOf(s) === -1; }); }
    var grid = $("#compare-grid");
    grid.textContent = "";
    function rowLabel(text) { grid.appendChild(el("div", { class: "cmp-label", text: text })); }

    rowLabel("");
    [A, B].forEach(function (c, i) {
      var other = i === 0 ? B : A;
      grid.appendChild(el("section", { class: "cmp-head compare-col", "aria-label": nameOf(c) }, [
        el("div", { class: "cmp-head-top" }, [
          el("div", { class: "cand" }, [el("span", { class: "avatar", "aria-hidden": "true", text: TL.initials(nameOf(c)) }),
            el("div", { class: "cand-text" }, [el("h3", { text: nameOf(c), title: nameOf(c) }), el("div", { class: "cand-file", text: "Rank #" + c.rank + " · " + c.filename })])]),
          el("button", { type: "button", class: "btn btn-ghost btn-icon btn-sm", "aria-label": "Remove " + nameOf(c) + " from comparison", "data-tip": "Remove",
            onclick: function () { removeFromCompare(keyOf(c)); } }, [TL.icon("x")])
        ]),
        el("div", { class: "cmp-big" }, [el("b", { text: c.score.toFixed(1) }), el("span", { class: "muted xs", text: c.breakdown.label || "" }),
          c.score > other.score + 0.05 ? el("span", { class: "delta", text: "+" + (c.score - other.score).toFixed(1) + " overall" }) : null])
      ]));
    });
    [["Text similarity", "tfidf_score", "w-tfidf"], ["Skill coverage", "skill_coverage", "w-skills"], ["Section signal", "section_signal", "w-sections"]].forEach(function (r) {
      rowLabel(r[0]);
      grid.appendChild(cmpValue(a[r[1]], b[r[1]], r[2]));
      grid.appendChild(cmpValue(b[r[1]], a[r[1]], r[2]));
    });
    rowLabel("Required skills matched");
    [A, B].forEach(function (c) {
      var bd = c.breakdown;
      grid.appendChild(el("div", { class: "cmp-val" }, [el("b", { text: (bd.matched_skills || []).length + " of " + (bd.required_skills || []).length })]));
    });
    rowLabel("Skills only this candidate has");
    [[a, b], [b, a]].forEach(function (xy) {
      var u = only(xy[0], xy[1]);
      grid.appendChild(el("div", {}, [u.length ? chipList(u, "chip-ok", 100, "check") : el("span", { class: "muted xs", text: "None" })]));
    });
    rowLabel("Shared skills (" + shared.length + ")");
    grid.appendChild(el("div", { class: "span-2" }, [shared.length ? chipList(shared, "chip-neutral", 100) : el("span", { class: "muted xs", text: "None" })]));
    rowLabel("Missing");
    [a, b].forEach(function (bd) {
      var m = bd.missing_skills || [];
      grid.appendChild(el("div", {}, [m.length ? chipList(m, "chip-missing", 100, "x") : el("span", { class: "muted xs", text: "None" })]));
    });
    rowLabel("Evidence");
    [a, b].forEach(function (bd) {
      var ev = snippetsOf(bd, 2);
      grid.appendChild(el("div", {}, ev.length ? ev : [el("span", { class: "muted xs", text: "No excerpts" })]));
    });
    rowLabel("Warnings");
    [a, b].forEach(function (bd) {
      var ws = bd.warnings || [];
      grid.appendChild(el("div", {}, [ws.length ? el("ul", { class: "warnings" }, ws.map(function (x) { return el("li", {}, [TL.icon("alert"), el("span", { text: x })]); })) : el("span", { class: "muted xs", text: "None" })]));
    });
    var dialog = $("#compare-dialog");
    if (!dialog.open) { if (dialog.showModal) dialog.showModal(); else dialog.setAttribute("open", ""); }
    $("#compare-close").focus();
  }
  function removeFromCompare(key) {
    compare = compare.filter(function (k) { return k !== key; });
    render();
    var dialog = $("#compare-dialog");
    if (dialog.open) dialog.close();
    $("#compare-btn").focus();
    TL.toast("Removed from comparison. Select another candidate to compare.");
  }
  $("#compare-btn").addEventListener("click", openCompare);
  $("#compare-close").addEventListener("click", function () { $("#compare-dialog").close(); $("#compare-btn").focus(); });

  /* ---------------------------------------------------------------- export + print */
  var CSV_HEADER = ["Candidate PDF", "Text similarity (%)", "Matched terms", "Terms not found", "Explanation", "Rank", "Final score",
    "Label", "Skill coverage (%)", "Section signal (%)", "Must-have missing", "Shortlisted", "Notes", "Warnings"];
  $("#export-btn").addEventListener("click", function () {
    var rows = visibleCandidates();
    var lines = [CSV_HEADER.map(TL.csvCell).join(",")];
    rows.forEach(function (c) {
      var b = c.breakdown;
      lines.push([c.filename, b.tfidf_score, (c.matched_terms || []).join(", "), (c.missing_terms || []).join(", "), c.explanation,
        c.rank, c.score, b.label, b.skill_coverage, b.section_signal, (b.must_have_missing || []).join(", "),
        c.shortlisted ? "yes" : "no", c.notes || "", (b.warnings || []).join(" | ")].map(TL.csvCell).join(","));
    });
    var slug = (data.job_title || "screening").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "screening";
    TL.downloadText("talentlens-" + slug + ".csv", lines.join("\r\n") + "\r\n");
    TL.toast("Exported " + rows.length + " candidate" + (rows.length === 1 ? "" : "s") + " (current view).");
  });
  $("#print-btn").addEventListener("click", function () { window.print(); });

  /* ---------------------------------------------------------------- delete with undo (owner, saved screenings) */
  var del = $("#delete-screening");
  if (del) {
    var pending = null;
    var sendDelete = function (keepalive) {
      return TL.api("/api/v1/screenings/" + screeningId, { method: "DELETE", keepalive: keepalive, retry: false });
    };
    window.addEventListener("pagehide", function () { if (pending) { clearTimeout(pending); pending = null; sendDelete(true); } });
    del.addEventListener("click", function () {
      $("#results-content").hidden = true;
      del.disabled = true;
      pending = setTimeout(function () {
        pending = null;
        sendDelete(false).then(function () { window.location.assign("/history"); })
          .catch(function (err) { $("#results-content").hidden = false; del.disabled = false; TL.toast(err.message, { type: "error" }); });
      }, 6500);
      TL.toast("Screening will be deleted.", { duration: 6500, action: { label: "Undo", onClick: function () {
        clearTimeout(pending); pending = null; $("#results-content").hidden = false; del.disabled = false; del.focus(); TL.toast("Deletion cancelled.");
      } } });
    });
  }

  showState("loading");
  start();
})();
