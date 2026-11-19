/* Results screen: summary, filters, sorting, explanations, shortlist, compare, CSV, print. */
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

  function keyOf(c) { return c.id ? "id-" + c.id : "f-" + c.filename; }
  function nameOf(c) { return c.display_name || c.filename.replace(/\.pdf$/i, "").replace(/[_-]+/g, " "); }
  function coverageText(b) {
    var total = (b.required_skills || []).length;
    if (b.skill_coverage === null || b.skill_coverage === undefined) return "n/a";
    if (!total) return "n/a (no skills in JD)";
    return (b.matched_skills || []).length + " of " + total + " (" + Math.round(b.skill_coverage) + "%)";
  }
  function pct(n) { return n === null || n === undefined ? "—" : Number(n).toFixed(1); }

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
    $("#results-subtitle").textContent = (data.saved ? "Saved " : "Screened ") + TL.formatDate(data.created_at) +
      " · scoring v" + (data.scoring_version || "2.0") + (data.saved ? "" : " · not stored on the server");
    document.title = (data.job_title || "Results") + " · TalentLens";
    $("#sample-banner").hidden = !data.sample_data;
    renderFileErrors();
    renderSummary();
    renderJob();
    render();
    $("#results-status").textContent = "Loaded " + data.candidates.length + " candidates.";
  }

  /* ---------------------------------------------------------------- summary + job */
  function renderFileErrors() {
    var box = $("#file-errors");
    var errs = data.file_errors || [];
    box.textContent = "";
    if (!errs.length) { box.hidden = true; return; }
    box.appendChild(el("strong", { text: errs.length + " file(s) were skipped:" }));
    box.appendChild(el("ul", {}, errs.map(function (e) { return el("li", { text: e.error }); })));
    box.hidden = false;
  }

  function renderSummary() {
    var cs = data.candidates;
    var avg = cs.length ? cs.reduce(function (s, c) { return s + c.score; }, 0) / cs.length : 0;
    var top = cs.reduce(function (m, c) { return Math.max(m, c.score); }, 0);
    $("#sum-count").textContent = String(cs.length);
    $("#sum-average").textContent = avg.toFixed(1);
    $("#sum-top").textContent = top.toFixed(1);
    var list = $("#sum-missing");
    list.textContent = "";
    var missing = (data.summary && data.summary.top_missing_skills) || [];
    if (!missing.length) list.appendChild(el("li", { class: "muted small", text: "No required skills are missing across the pool." }));
    missing.forEach(function (m) {
      list.appendChild(el("li", { class: "chip chip-neutral", text: m.skill + " — missing in " + m.missing_count + " of " + cs.length }));
    });
  }

  function renderJob() {
    var w = data.weights || (data.candidates[0] && data.candidates[0].breakdown.weights) || {};
    $("#weights-text").textContent = w.tfidf !== undefined
      ? "Weights: text similarity " + Math.round(w.tfidf * 100) + "%, skill coverage " + Math.round((w.skills || 0) * 100) +
        "%, section signal " + Math.round((w.sections || 0) * 100) + "%." + (data.must_have && data.must_have.length ? " Must-haves: " + data.must_have.join(", ") + "." : "")
      : "Imported from v1.x: scores are text similarity only.";
    var req = data.required_skills || [];
    $("#required-count").textContent = String(req.length);
    var list = $("#required-skills");
    list.textContent = "";
    req.forEach(function (s) { list.appendChild(el("li", { class: "chip chip-neutral", text: s + ((data.must_have || []).indexOf(s) !== -1 ? " (must-have)" : "") })); });
    $("#job-description-text").textContent = data.job_description || "";
  }

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

  /* ---------------------------------------------------------------- rendering */
  function scoreCell(c) {
    var b = c.breakdown;
    var fill = el("span");
    fill.style.width = Math.max(0, Math.min(100, c.score)) + "%";
    return el("div", { class: "score", dataset: { band: TL.band(c.score) } }, [
      el("span", { class: "score-value", text: c.score.toFixed(1) }),
      el("span", { class: "score-bar", "aria-hidden": "true" }, [fill]),
      el("span", { class: "score-label", text: (b.label || "") + (b.capped ? " · capped" : "") })
    ]);
  }

  function chipList(items, cls, limit, prefix) {
    var shown = items.slice(0, limit);
    var ul = el("ul", { class: "chips" }, shown.map(function (s) { return el("li", { class: "chip " + cls }, [prefix ? el("span", { "aria-hidden": "true", text: prefix }) : null, s]); }));
    if (items.length > limit) ul.appendChild(el("li", { class: "chip chip-neutral", text: "+" + (items.length - limit) + " more" }));
    return ul;
  }

  function flags(c) {
    var b = c.breakdown, out = [];
    if (c.shortlisted) out.push(el("span", { class: "badge badge-accent", text: "Shortlisted" }));
    if (b.low_text) out.push(el("span", { class: "badge badge-warn", text: "Likely scanned" }));
    if (b.capped || (b.must_have_missing && b.must_have_missing.length)) out.push(el("span", { class: "badge badge-danger", text: "Missing must-have" }));
    return out.length ? el("div", { class: "cand-flags" }, out) : null;
  }

  function starButton(c) {
    var btn = el("button", {
      type: "button", class: "btn btn-ghost btn-icon star-btn", "aria-pressed": c.shortlisted ? "true" : "false",
      "aria-label": (c.shortlisted ? "Remove " : "Add ") + nameOf(c) + (c.shortlisted ? " from shortlist" : " to shortlist"),
      title: c.shortlisted ? "Shortlisted" : "Shortlist",
      onclick: function () { toggleShortlist(c); }
    }, [TL.svg(c.shortlisted ? TL.icons.star : TL.icons.starOutline)]);
    return btn;
  }

  function compareBox(c) {
    var key = keyOf(c);
    var checked = compare.indexOf(key) !== -1;
    var input = el("input", { type: "checkbox", checked: checked, disabled: !checked && compare.length >= 2,
      "aria-label": "Select " + nameOf(c) + " for comparison", onchange: function (e) {
        if (e.target.checked) { if (compare.length < 2) compare.push(key); }
        else compare = compare.filter(function (k) { return k !== key; });
        render();
        var again = document.getElementById("cmp-" + cssId(key));
        if (again) again.focus();
      } });
    input.id = "cmp-" + cssId(key);
    return el("label", { class: "compare-check" }, [input, "Compare"]);
  }

  function cssId(key) { return key.replace(/[^a-zA-Z0-9_-]/g, "_"); }

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
      var whyBtn = el("button", { type: "button", class: "btn btn-secondary btn-sm", "aria-expanded": isOpen ? "true" : "false",
        "aria-controls": detailId, text: isOpen ? "Hide details" : "Why this score?", onclick: function () {
          expanded[key] = !expanded[key]; render();
          var again = document.querySelector('[aria-controls="' + detailId + '"]');
          if (again) again.focus();
        } });
      body.appendChild(el("tr", { class: c.shortlisted ? "is-shortlisted" : "" }, [
        el("td", { class: "col-rank", "data-label": "Rank", text: "#" + c.rank }),
        el("td", { "data-label": "Candidate" }, [el("div", { class: "cand-name", text: nameOf(c) }), el("div", { class: "cand-file", text: c.filename }), flags(c)]),
        el("td", { "data-label": "Score" }, [scoreCell(c)]),
        el("td", { "data-label": "Skill coverage", text: coverageText(b) }),
        el("td", { "data-label": "Skills" }, [
          el("div", { class: "small muted", text: "Matched (" + (b.matched_skills || []).length + ")" }),
          chipList(b.matched_skills || [], "chip-ok", 6, "✓ "),
          (b.missing_skills || []).length ? el("div", { class: "small muted mt-2", text: "Missing (" + b.missing_skills.length + ")" }) : null,
          (b.missing_skills || []).length ? chipList(b.missing_skills, "chip-missing", 4, "✗ ") : null
        ]),
        el("td", { "data-label": "Actions" }, [el("div", { class: "row-actions" }, [starButton(c), compareBox(c), whyBtn])])
      ]));
      var detailRow = el("tr", { class: "detail-row", id: detailId, hidden: !isOpen }, [el("td", { colspan: "6" }, [isOpen ? detailPanel(c) : null])]);
      body.appendChild(detailRow);
    });
    var total = data.candidates.length;
    $("#result-count").textContent = "Showing " + rows.length + " of " + total + " candidate" + (total === 1 ? "" : "s") + ".";
    $("#filter-empty").hidden = rows.length > 0;
    $("#results-table").hidden = rows.length === 0;
    var cmp = $("#compare-btn");
    cmp.textContent = "Compare (" + compare.length + "/2)";
    cmp.disabled = compare.length !== 2;
  }

  function detailPanel(c) {
    var b = c.breakdown;
    var w = b.weights || { tfidf: 1, skills: 0, sections: 0 };
    var contrib = b.contributions || {};
    var legacy = b.skill_coverage === null || b.skill_coverage === undefined;
    var left = el("div", {}, [
      el("h3", { text: "Score breakdown" }),
      legacy ? el("p", { class: "small", text: "Imported from v1.x: this score is TF-IDF text similarity only." }) :
        el("p", { class: "formula", text: "score " + c.score.toFixed(1) + " = " +
          Math.round(w.tfidf * 100) + "% × " + pct(b.tfidf_score) + " (text) + " +
          Math.round(w.skills * 100) + "% × " + pct(b.skill_coverage) + " (skills) + " +
          Math.round(w.sections * 100) + "% × " + pct(b.section_signal) + " (sections)" + (b.capped ? " → capped" : "") }),
      legacy ? null : el("table", { class: "breakdown" }, [
        el("thead", {}, [el("tr", {}, [el("th", { scope: "col", text: "Component" }), el("th", { scope: "col", text: "Value" }), el("th", { scope: "col", text: "Weight" }), el("th", { scope: "col", text: "Points" })])]),
        el("tbody", {}, [
          row("Text similarity (TF-IDF)", b.tfidf_score, w.tfidf, contrib.tfidf),
          row("Skill coverage", b.skill_coverage, w.skills, contrib.skills),
          row("Section signal", b.section_signal, w.sections, contrib.sections)
        ])
      ]),
      b.warnings && b.warnings.length ? el("div", { class: "notice notice-warn small mt-4" }, [el("strong", { text: "Warnings" }), el("ul", { class: "warnings" }, b.warnings.map(function (x) { return el("li", { text: x }); }))]) : null,
      el("h3", { class: "mt-4", text: "Top shared terms" }),
      (b.top_terms || []).length ? chipList(b.top_terms, "chip-neutral", 10) : el("p", { class: "small muted", text: "No shared terms." }),
      el("h3", { class: "mt-4", text: "All matched skills" }),
      (b.matched_skills || []).length ? chipList(b.matched_skills, "chip-ok", 100, "✓ ") : el("p", { class: "small muted", text: "None." }),
      el("h3", { class: "mt-4", text: "All missing skills" }),
      (b.missing_skills || []).length ? chipList(b.missing_skills, "chip-missing", 100, "✗ ") : el("p", { class: "small muted", text: "None." })
    ]);
    var snippets = (b.snippets || []).map(function (s) {
      var p = el("p", { class: "snippet" });
      if (s.prefix_ellipsis) p.appendChild(document.createTextNode("…"));
      s.segments.forEach(function (seg) { p.appendChild(seg.mark ? el("mark", { text: seg.text }) : document.createTextNode(seg.text)); });
      if (s.suffix_ellipsis) p.appendChild(document.createTextNode("…"));
      return p;
    });
    var right = el("div", {}, [
      el("h3", { text: "Where matched skills appear" }),
      snippets.length ? el("div", {}, snippets) : el("p", { class: "small muted", text: "No excerpts available." }),
      (b.skills_in_sections || []).length ? el("p", { class: "small muted", text: "Found inside Skills / Experience / Projects sections: " + b.skills_in_sections.join(", ") + "." }) : null,
      el("p", { class: "small muted", text: (b.text_chars || 0).toLocaleString() + " characters of text were extracted." }),
      notesBox(c)
    ]);
    return el("div", { class: "detail" }, [left, right]);
  }

  function row(label, value, weight, points) {
    return el("tr", {}, [el("th", { scope: "row", text: label }), el("td", { class: "num", text: pct(value) }),
      el("td", { class: "num", text: Math.round((weight || 0) * 100) + "%" }), el("td", { class: "num", text: pct(points) })]);
  }

  function notesBox(c) {
    if (!(screeningId && isOwner && c.id)) return null;
    var id = "notes-" + c.id;
    var area = el("textarea", { class: "textarea", id: id, maxlength: "5000" });
    area.value = c.notes || "";
    var save = el("button", { type: "button", class: "btn btn-secondary btn-sm mt-2", text: "Save note", onclick: function () {
      save.disabled = true;
      TL.api("/api/v1/screenings/" + screeningId + "/candidates/" + c.id, { method: "PATCH", json: { notes: area.value } })
        .then(function (updated) { c.notes = updated.notes; TL.toast("Note saved."); })
        .catch(function (err) { TL.toast(err.message, { type: "error" }); })
        .then(function () { save.disabled = false; });
    } });
    return el("div", { class: "field notes mt-4" }, [el("label", { for: id, text: "Private notes" }), area, save]);
  }

  /* ---------------------------------------------------------------- actions */
  function toggleShortlist(c) {
    var next = !c.shortlisted;
    c.shortlisted = next;
    render();
    var focusTarget = function () { var b = TL.$$(".star-btn").filter(function (x) { return x.getAttribute("aria-label").indexOf(nameOf(c)) !== -1; })[0]; if (b) b.focus(); };
    focusTarget();
    if (screeningId && c.id) {
      TL.api("/api/v1/screenings/" + screeningId + "/candidates/" + c.id, { method: "PATCH", json: { shortlisted: next } })
        .then(function () { TL.toast(next ? nameOf(c) + " added to the shortlist." : nameOf(c) + " removed from the shortlist."); })
        .catch(function (err) { c.shortlisted = !next; render(); focusTarget(); TL.toast(err.message, { type: "error" }); });
    } else {
      persistLocal();
      TL.toast(next ? nameOf(c) + " shortlisted (this tab only)." : nameOf(c) + " removed from the shortlist.");
    }
  }

  function openCompare() {
    var pair = compare.map(function (k) { return data.candidates.filter(function (c) { return keyOf(c) === k; })[0]; }).filter(Boolean);
    if (pair.length !== 2) return;
    var a = pair[0].breakdown, b = pair[1].breakdown;
    var onlyA = (a.matched_skills || []).filter(function (s) { return (b.matched_skills || []).indexOf(s) === -1; });
    var onlyB = (b.matched_skills || []).filter(function (s) { return (a.matched_skills || []).indexOf(s) === -1; });
    var grid = $("#compare-grid");
    grid.textContent = "";
    pair.forEach(function (c, i) {
      var bd = c.breakdown;
      var unique = i === 0 ? onlyA : onlyB;
      grid.appendChild(el("section", { class: "compare-col", "aria-label": nameOf(c) }, [
        el("h3", { text: "#" + c.rank + " " + nameOf(c) }),
        scoreCell(c),
        el("table", { class: "breakdown mt-4" }, [el("tbody", {}, [
          row("Text similarity", bd.tfidf_score, (bd.weights || {}).tfidf, (bd.contributions || {}).tfidf),
          row("Skill coverage", bd.skill_coverage, (bd.weights || {}).skills, (bd.contributions || {}).skills),
          row("Section signal", bd.section_signal, (bd.weights || {}).sections, (bd.contributions || {}).sections)
        ])]),
        el("h4", { class: "mt-4", text: "Skills only this candidate has (" + unique.length + ")" }),
        unique.length ? chipList(unique, "chip-ok", 100, "✓ ") : el("p", { class: "small muted", text: "None." }),
        el("h4", { class: "mt-4", text: "Missing (" + (bd.missing_skills || []).length + ")" }),
        (bd.missing_skills || []).length ? chipList(bd.missing_skills, "chip-missing", 100, "✗ ") : el("p", { class: "small muted", text: "None." }),
        bd.warnings && bd.warnings.length ? el("ul", { class: "warnings small mt-4" }, bd.warnings.map(function (x) { return el("li", { text: x }); })) : null
      ]));
    });
    var dialog = $("#compare-dialog");
    if (dialog.showModal) dialog.showModal(); else dialog.setAttribute("open", "");
    $("#compare-close").focus();
  }
  $("#compare-btn").addEventListener("click", openCompare);
  $("#compare-close").addEventListener("click", function () { $("#compare-dialog").close(); $("#compare-btn").focus(); });

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

  /* delete with undo (owner, saved screenings) */
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
