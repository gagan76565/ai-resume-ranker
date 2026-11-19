/* Upload screen: drag-and-drop, per-file validation, must-have tags, progress. */
(function () {
  "use strict";
  var TL = window.TL, $ = TL.$, el = TL.el;

  var limits = { max_file_mb: 5, max_files: 20, max_total_mb: 30 };
  var state = { files: [], useSamples: false, samples: null, tags: [], nextId: 1, busy: false };
  var MAX_TAGS = 20;

  var form = $("#screen-form");
  var fileInput = $("#file-input");
  var dropzone = $("#dropzone");
  var fileList = $("#file-list");
  var fileStatus = $("#file-status");
  var jd = $("#job-description");
  var jdCounter = $("#jd-counter");
  var jdError = $("#jd-error");
  var formError = $("#form-error");
  var tagInput = $("#must-have-input");
  var tagList = $("#must-have-tags");
  var submitBtn = $("#submit-btn");

  /* ---------------------------------------------------------------- config + skills */
  TL.api("/api/v1/config").then(function (cfg) {
    limits = cfg.limits;
    $("#cap-value").textContent = String(cfg.must_have_cap);
  }).catch(function () { /* defaults are fine */ });

  TL.api("/api/v1/skills").then(function (data) {
    var list = $("#skill-options");
    data.skills.forEach(function (s) { list.appendChild(el("option", { value: s.canonical })); });
  }).catch(function () { /* autocomplete is optional */ });

  /* ---------------------------------------------------------------- job description */
  function updateCounter() {
    var n = jd.value.length;
    jdCounter.textContent = n.toLocaleString() + " / 20,000";
    jdCounter.classList.toggle("is-bad", n > 0 && n < 20);
  }
  jd.addEventListener("input", function () { updateCounter(); if (jd.value.trim().length >= 20) setJdError(""); });
  function setJdError(message) {
    jdError.textContent = message;
    jdError.hidden = !message;
    jd.setAttribute("aria-invalid", message ? "true" : "false");
  }

  /* ---------------------------------------------------------------- tags */
  function renderTags() {
    tagList.textContent = "";
    state.tags.forEach(function (tag, index) {
      tagList.appendChild(el("li", { class: "tag" }, [
        tag,
        el("button", { type: "button", "aria-label": "Remove must-have " + tag, onclick: function () {
          state.tags.splice(index, 1); renderTags(); tagInput.focus();
        } }, [TL.svg(TL.icons.close)])
      ]));
    });
    tagInput.disabled = state.tags.length >= MAX_TAGS;
    tagInput.placeholder = state.tags.length >= MAX_TAGS ? "Limit of 20 reached" : "Type a skill and press Enter";
  }
  function addTag(raw) {
    var value = raw.replace(/\s+/g, " ").trim().slice(0, 60);
    if (!value) return;
    var exists = state.tags.some(function (t) { return t.toLowerCase() === value.toLowerCase(); });
    if (!exists && state.tags.length < MAX_TAGS) state.tags.push(value);
    renderTags();
  }
  tagInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      addTag(tagInput.value); tagInput.value = "";
    } else if (event.key === "Backspace" && !tagInput.value && state.tags.length) {
      state.tags.pop(); renderTags();
    }
  });
  tagInput.addEventListener("change", function () {
    // Picking from the datalist fires "change"; add it straight away.
    if (tagInput.value && $("#skill-options").querySelector('option[value="' + CSS.escape(tagInput.value) + '"]')) {
      addTag(tagInput.value); tagInput.value = "";
    }
  });
  $("#must-have-box").addEventListener("click", function (e) { if (e.target === this) tagInput.focus(); });

  /* ---------------------------------------------------------------- files */
  function totalOkBytes() {
    return state.files.reduce(function (sum, f) { return f.status === "ok" ? sum + f.file.size : sum; }, 0);
  }
  function validateFile(entry) {
    var f = entry.file;
    if (!/\.pdf$/i.test(f.name)) return Promise.resolve(f.name + " is not a PDF file.");
    if (f.size === 0) return Promise.resolve(f.name + " is empty.");
    if (f.size > limits.max_file_mb * 1024 * 1024) return Promise.resolve(f.name + " is larger than " + limits.max_file_mb + " MB.");
    var head = f.slice(0, 5);
    var reader = head.arrayBuffer ? head.arrayBuffer() : Promise.resolve(null);
    return reader.then(function (buf) {
      if (!buf) return null;
      var bytes = new Uint8Array(buf);
      var magic = String.fromCharCode.apply(null, bytes);
      return magic === "%PDF-" ? null : f.name + " does not look like a real PDF (missing %PDF header).";
    }).catch(function () { return null; });
  }

  function addFiles(list) {
    if (state.useSamples) clearSamples();
    var incoming = Array.prototype.slice.call(list || []);
    incoming.forEach(function (file) {
      var duplicate = state.files.some(function (e) { return e.file.name === file.name && e.file.size === file.size; });
      if (duplicate) return;
      var entry = { id: state.nextId++, file: file, status: "checking", error: "" };
      if (state.files.length >= limits.max_files) {
        entry.status = "error"; entry.error = "Over the " + limits.max_files + "-file limit; remove a file first.";
        state.files.push(entry); return;
      }
      state.files.push(entry);
      validateFile(entry).then(function (problem) {
        if (problem) { entry.status = "error"; entry.error = problem; }
        else if (totalOkBytes() + file.size > limits.max_total_mb * 1024 * 1024) {
          entry.status = "error"; entry.error = "Would exceed the " + limits.max_total_mb + " MB total upload limit.";
        } else entry.status = "ok";
        renderFiles();
      });
    });
    renderFiles();
  }

  function removeFile(id) {
    state.files = state.files.filter(function (e) { return e.id !== id; });
    renderFiles();
    $("#browse-btn").focus();
  }

  function renderFiles() {
    fileList.textContent = "";
    if (state.useSamples && state.samples) {
      state.samples.resumes.forEach(function (r) {
        fileList.appendChild(el("li", { class: "file-chip" }, [
          TL.svg(TL.icons.file),
          el("div", {}, [el("div", { class: "file-name", text: r.filename }), el("div", { class: "file-meta", text: "Synthetic sample · " + r.description })]),
          el("span", { class: "badge badge-accent", text: "Sample" })
        ]));
      });
      fileStatus.textContent = state.samples.resumes.length + " synthetic sample resumes selected.";
      return;
    }
    state.files.forEach(function (entry) {
      var meta = entry.status === "checking" ? "Checking…" : entry.status === "error" ? entry.error : TL.formatBytes(entry.file.size) + " · Ready";
      var icon = TL.svg(TL.icons.file);
      icon.setAttribute("class", "file-icon");
      fileList.appendChild(el("li", { class: "file-chip" + (entry.status === "error" ? " is-invalid" : "") }, [
        icon,
        el("div", {}, [
          el("div", { class: "file-name", text: entry.file.name, title: entry.file.name }),
          el("div", { class: "file-meta" + (entry.status === "error" ? " is-bad" : ""), text: (entry.status === "error" ? "Problem: " : "") + meta })
        ]),
        el("button", { type: "button", class: "btn btn-ghost btn-icon", "aria-label": "Remove " + entry.file.name, onclick: function () { removeFile(entry.id); } }, [TL.svg(TL.icons.close)])
      ]));
    });
    var ok = state.files.filter(function (e) { return e.status === "ok"; }).length;
    var bad = state.files.filter(function (e) { return e.status === "error"; }).length;
    fileStatus.textContent = state.files.length === 0 ? "No files selected." :
      ok + " file" + (ok === 1 ? "" : "s") + " ready (" + TL.formatBytes(totalOkBytes()) + ")" + (bad ? ", " + bad + " with problems (they will be skipped)" : "") + ".";
  }

  $("#browse-btn").addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { addFiles(fileInput.files); fileInput.value = ""; });
  ["dragenter", "dragover"].forEach(function (name) {
    dropzone.addEventListener(name, function (e) { e.preventDefault(); dropzone.classList.add("is-dragover"); });
  });
  ["dragleave", "dragend", "drop"].forEach(function (name) {
    dropzone.addEventListener(name, function (e) { e.preventDefault(); dropzone.classList.remove("is-dragover"); });
  });
  dropzone.addEventListener("drop", function (e) { if (e.dataTransfer) addFiles(e.dataTransfer.files); });
  // Prevent the browser from opening a PDF dropped outside the zone.
  window.addEventListener("dragover", function (e) { e.preventDefault(); });
  window.addEventListener("drop", function (e) { e.preventDefault(); });

  /* ---------------------------------------------------------------- samples */
  function loadSamples() {
    if (state.samples) return Promise.resolve(state.samples);
    return TL.api("/api/v1/samples").then(function (data) { state.samples = data; return data; });
  }
  function fillSampleJob(data) {
    $("#job-title").value = data.job_title;
    jd.value = data.job_description;
    updateCounter(); setJdError("");
    state.tags = data.must_have.slice(); renderTags();
  }
  $("#load-sample-jd").addEventListener("click", function () {
    loadSamples().then(function (data) { fillSampleJob(data); TL.toast("Sample job description loaded (synthetic demo data)."); jd.focus(); })
      .catch(function (err) { TL.toast(err.message, { type: "error" }); });
  });
  $("#try-samples").addEventListener("click", function () {
    loadSamples().then(function (data) {
      fillSampleJob(data);
      state.useSamples = true;
      state.files = [];
      $("#sample-notice").hidden = false;
      renderFiles();
      TL.toast("Loaded 6 synthetic sample resumes. Press “Rank candidates” to see the results.");
      submitBtn.focus();
    }).catch(function (err) { TL.toast(err.message, { type: "error" }); });
  });
  function clearSamples() {
    state.useSamples = false;
    $("#sample-notice").hidden = true;
    renderFiles();
  }
  $("#clear-samples").addEventListener("click", function () { clearSamples(); $("#browse-btn").focus(); });

  /* ---------------------------------------------------------------- submit */
  function showFormError(message, details) {
    formError.textContent = "";
    formError.appendChild(el("strong", { text: message }));
    if (details && details.length) {
      formError.appendChild(el("ul", {}, details.map(function (d) { return el("li", { text: d.error || d.message || String(d) }); })));
    }
    formError.hidden = false;
    formError.focus && formError.setAttribute("tabindex", "-1");
    formError.focus();
  }
  function setBusy(busy, text) {
    state.busy = busy;
    submitBtn.disabled = busy;
    submitBtn.setAttribute("aria-busy", busy ? "true" : "false");
    $("#submit-label").textContent = busy ? "Ranking…" : "Rank candidates";
    $("#progress-wrap").hidden = !busy;
    if (text) $("#progress-text").textContent = text;
  }
  function setProgress(pct) {
    var bar = $("#progress");
    if (pct === null) { bar.classList.add("is-indeterminate"); bar.removeAttribute("aria-valuenow"); $("#progress-pct").textContent = ""; return; }
    bar.classList.remove("is-indeterminate");
    bar.setAttribute("aria-valuenow", String(pct));
    bar.firstElementChild.style.width = pct + "%";
    $("#progress-pct").textContent = pct + "%";
  }

  function saveResult(data) {
    if (TL.store.set("sessionStorage", "tl:last-result", data)) return;
    try { window.name = "tl:" + JSON.stringify(data); } catch (e) { /* ignore */ }
  }

  function send(formData, attempt) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/v1/screen");
    xhr.setRequestHeader("X-CSRF-Token", TL.csrf());
    xhr.setRequestHeader("Accept", "application/json");
    xhr.responseType = "text";
    xhr.upload.addEventListener("progress", function (e) {
      if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
    });
    xhr.upload.addEventListener("load", function () { $("#progress-text").textContent = "Scoring resumes…"; setProgress(null); });
    xhr.addEventListener("load", function () {
      var data = {};
      try { data = JSON.parse(xhr.responseText || "{}"); } catch (e) { data = {}; }
      if ((xhr.status === 502 || xhr.status === 503 || xhr.status === 504) && attempt < 3) return retry(formData, attempt);
      if (xhr.status >= 200 && xhr.status < 300) {
        $("#progress-text").textContent = "Done. Opening results…";
        saveResult(data);
        window.location.assign(data.saved && data.screening_id ? "/screenings/" + data.screening_id : "/results");
        return;
      }
      setBusy(false);
      var err = data.error || {};
      showFormError(err.message || "Screening failed (HTTP " + xhr.status + ").", err.details && err.code === "no_readable_resumes" ? err.details : null);
    });
    xhr.addEventListener("error", function () {
      if (attempt < 3) return retry(formData, attempt);
      setBusy(false);
      showFormError("Could not reach the server. Check your connection and try again.");
    });
    xhr.send(formData);
  }
  function retry(formData, attempt) {
    var wait = [5, 10, 20][attempt];
    $("#progress-text").textContent = "The free server is waking up. Retrying in " + wait + " seconds…";
    setProgress(null);
    setTimeout(function () { send(formData, attempt + 1); }, wait * 1000);
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (state.busy) return;
    formError.hidden = true;
    if (tagInput.value.trim()) { addTag(tagInput.value); tagInput.value = ""; }
    var text = jd.value.trim();
    if (text.length < 20) { setJdError("Enter a job description of at least 20 characters."); jd.focus(); return; }
    var ready = state.files.filter(function (e) { return e.status === "ok"; });
    if (!state.useSamples) {
      if (state.files.some(function (e) { return e.status === "checking"; })) { showFormError("Still checking files. Try again in a moment."); return; }
      if (!ready.length) { showFormError("Add at least one valid PDF resume, or press “Try with sample resumes”."); $("#browse-btn").focus(); return; }
    }
    var fd = new FormData();
    fd.append("job_title", $("#job-title").value.trim());
    fd.append("job_description", text);
    state.tags.forEach(function (t) { fd.append("must_have", t); });
    var save = $("#save-toggle");
    fd.append("save", save && !save.checked ? "0" : "1");
    if (state.useSamples) fd.append("use_samples", "1");
    else ready.forEach(function (e) { fd.append("resumes", e.file, e.file.name); });
    setBusy(true, state.useSamples ? "Scoring sample resumes…" : "Uploading…");
    setProgress(state.useSamples ? null : 0);
    send(fd, 0);
  });

  updateCounter();
  renderTags();
  renderFiles();
})();
