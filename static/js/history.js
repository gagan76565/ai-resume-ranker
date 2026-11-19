/* Owner history: search, open, delete with undo. */
(function () {
  "use strict";
  var TL = window.TL, $ = TL.$, el = TL.el;
  var items = [];
  var hiddenIds = {};
  var pendingDeletes = {};
  var searchTimer = null;

  function load(q) {
    $("#h-loading").hidden = false;
    TL.api("/api/v1/screenings" + (q ? "?q=" + encodeURIComponent(q) : "")).then(function (data) {
      items = data.screenings;
      render(q);
    }).catch(function (err) {
      items = [];
      render(q);
      $("#h-empty-title").textContent = "Could not load history";
      $("#h-empty-text").textContent = err.message;
    }).then(function () { $("#h-loading").hidden = true; });
  }

  function render(q) {
    var list = $("#h-list");
    list.textContent = "";
    var visible = items.filter(function (s) { return !hiddenIds[s.id]; });
    visible.forEach(function (s) {
      var link = el("a", { href: "/screenings/" + s.id, text: s.job_title || "Untitled role" });
      list.appendChild(el("li", { class: "history-item" }, [
        el("span", { class: "icon-tile" }, [TL.icon("briefcase")]),
        el("div", { class: "cand-text" }, [
          el("h3", {}, [link]),
          el("div", { class: "history-meta" }, [
            el("span", { text: TL.formatDate(s.created_at) }),
            el("span", { text: s.candidate_count + " candidate" + (s.candidate_count === 1 ? "" : "s") }),
            s.average_score !== null ? el("span", { text: "Average " + s.average_score.toFixed(1) }) : null,
            s.shortlisted_count ? el("span", { class: "badge badge-accent" }, [TL.icon("star-fill"), s.shortlisted_count + " shortlisted"]) : null,
            s.scoring_version === "1.x-import" ? el("span", { class: "badge badge-warn", text: "Imported v1" }) : null
          ])
        ]),
        s.top_score !== null ? el("div", { class: "history-score" }, [el("b", { text: s.top_score.toFixed(1) }), el("span", { text: "Top score" })]) : el("div"),
        el("div", { class: "btn-row" }, [
          el("a", { class: "btn btn-secondary btn-sm", href: "/screenings/" + s.id, "aria-label": "Open " + (s.job_title || "screening") }, ["Open", TL.icon("chevron-right")]),
          el("button", { type: "button", class: "btn btn-ghost btn-sm", "aria-label": "Delete " + (s.job_title || "screening"),
            onclick: function () { remove(s); } }, [TL.icon("trash"), "Delete"])
        ])
      ]));
    });
    $("#h-count").textContent = q ? visible.length + " result(s) for “" + q + "”" : visible.length + " saved screening" + (visible.length === 1 ? "" : "s");
    var empty = visible.length === 0;
    $("#h-empty").hidden = !empty;
    if (empty) {
      $("#h-empty-title").textContent = q ? "No screenings match your search" : "No saved screenings yet";
      $("#h-empty-text").textContent = q ? "Try a different word from the role title or job description." : "Screenings you run while signed in are saved here.";
    }
  }

  function sendDelete(id, keepalive) {
    return TL.api("/api/v1/screenings/" + id, { method: "DELETE", keepalive: keepalive, retry: false });
  }

  function remove(s) {
    hiddenIds[s.id] = true;
    render($("#h-search").value.trim());
    pendingDeletes[s.id] = setTimeout(function () {
      delete pendingDeletes[s.id];
      sendDelete(s.id, false).then(function () {
        items = items.filter(function (x) { return x.id !== s.id; });
      }).catch(function (err) {
        delete hiddenIds[s.id]; render($("#h-search").value.trim());
        TL.toast(err.message, { type: "error" });
      });
    }, 6500);
    TL.toast("Deleted “" + (s.job_title || "screening") + "”.", { duration: 6500, action: { label: "Undo", onClick: function () {
      clearTimeout(pendingDeletes[s.id]); delete pendingDeletes[s.id]; delete hiddenIds[s.id];
      render($("#h-search").value.trim()); TL.toast("Restored.");
    } } });
    $("#h-search").focus();
  }

  window.addEventListener("pagehide", function () {
    Object.keys(pendingDeletes).forEach(function (id) { clearTimeout(pendingDeletes[id]); sendDelete(id, true); });
    pendingDeletes = {};
  });

  $("#h-search").addEventListener("input", function (e) {
    clearTimeout(searchTimer);
    var q = e.target.value.trim();
    searchTimer = setTimeout(function () { load(q); }, 250);
  });

  load("");
})();
