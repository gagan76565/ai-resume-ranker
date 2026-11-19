/* TalentLens shared helpers. No framework, no build step.
   Security: every piece of user or resume text is inserted with textContent
   (via el()), never innerHTML. */
(function () {
  "use strict";

  var TL = (window.TL = window.TL || {});

  /* ---------------------------------------------------------------- DOM */
  TL.$ = function (sel, root) { return (root || document).querySelector(sel); };
  TL.$$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  /** el("div", {class: "x", "aria-label": "y", onclick: fn}, [children...]) — text children become text nodes. */
  TL.el = function (tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key.indexOf("on") === 0 && typeof value === "function") node.addEventListener(key.slice(2), value);
        else if (key === "dataset") Object.keys(value).forEach(function (k) { node.dataset[k] = value[k]; });
        else if (value === true) node.setAttribute(key, "");
        else node.setAttribute(key, String(value));
      });
    }
    (children || []).forEach(function (child) {
      if (child === null || child === undefined || child === false) return;
      node.appendChild(typeof child === "string" || typeof child === "number" ? document.createTextNode(String(child)) : child);
    });
    return node;
  };

  /** Icon from the inline sprite in templates/_icons.html: TL.icon("check", "extra-class"). */
  TL.icon = function (name, cls) {
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    if (cls) svg.setAttribute("class", cls);
    var use = document.createElementNS(ns, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  };

  /** Two-letter initials for an avatar ("sample 01 aarav placeholder" -> "SP"). */
  TL.initials = function (name) {
    var words = String(name || "").split(/\s+/).filter(function (w) { return /^[a-z]/i.test(w); });
    if (!words.length) return "#";
    var last = words.length > 1 ? words[words.length - 1][0] : (words[0][1] || "");
    return (words[0][0] + last).toUpperCase();
  };

  /** Accessible tabs: [role=tab][aria-controls] inside root. Arrow keys, Home/End. */
  TL.tabs = function (root) {
    var tabs = TL.$$('[role="tab"]', root);
    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !on;
      });
      if (focus) tab.focus();
    }
    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(tab); });
      tab.addEventListener("keydown", function (e) {
        var next = null;
        if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
        else if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
        else if (e.key === "Home") next = tabs[0];
        else if (e.key === "End") next = tabs[tabs.length - 1];
        if (next) { e.preventDefault(); select(next, true); }
      });
    });
    if (tabs.length) select(tabs.filter(function (t) { return t.getAttribute("aria-selected") === "true"; })[0] || tabs[0]);
  };

  /** Set a bar width in %, optionally animating from 0 (CSS transition; disabled under reduced motion). */
  TL.setWidth = function (node, pct, animate) {
    var value = Math.max(0, Math.min(100, Number(pct) || 0)) + "%";
    if (!animate) { node.style.width = value; return; }
    node.style.width = "0%";
    requestAnimationFrame(function () { requestAnimationFrame(function () { node.style.width = value; }); });
  };

  TL.formatBytes = function (bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  };
  TL.formatDate = function (iso) {
    if (!iso) return "";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  };
  TL.band = function (score) {
    if (score >= 70) return "strong";
    if (score >= 45) return "partial";
    if (score >= 20) return "weak";
    return "very-weak";
  };

  /* ---------------------------------------------------------------- storage (never required) */
  TL.store = {
    get: function (area, key) { try { var v = window[area].getItem(key); return v ? JSON.parse(v) : null; } catch (e) { return null; } },
    set: function (area, key, value) { try { window[area].setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; } },
    remove: function (area, key) { try { window[area].removeItem(key); } catch (e) { /* ignore */ } }
  };

  /* ---------------------------------------------------------------- toasts */
  TL.toast = function (message, opts) {
    opts = opts || {};
    var region = TL.$("#toast-region");
    if (!region) return;
    var toast = TL.el("div", { class: "toast" + (opts.type === "error" ? " toast-error" : ""), role: opts.type === "error" ? "alert" : "status" },
      [TL.icon(opts.type === "error" ? "alert" : "check", "toast-icon"), TL.el("span", { class: "toast-msg", text: message })]);
    var timer;
    function close() { clearTimeout(timer); if (toast.parentNode) toast.parentNode.removeChild(toast); if (opts.onClose) opts.onClose(); }
    if (opts.action) {
      toast.appendChild(TL.el("button", { type: "button", class: "btn btn-sm", text: opts.action.label, onclick: function () {
        opts.onClose = null; close(); opts.action.onClick();
      } }));
    }
    toast.appendChild(TL.el("button", { type: "button", class: "btn btn-sm btn-icon", "aria-label": "Dismiss notification", onclick: close }, [TL.icon("x")]));
    region.appendChild(toast);
    timer = setTimeout(close, opts.duration || (opts.action ? 6500 : 4500));
    return { close: close };
  };

  /* ---------------------------------------------------------------- API + cold-start handling */
  TL.csrf = function () { var m = TL.$('meta[name="csrf-token"]'); return m ? m.getAttribute("content") : ""; };

  var wake = { banner: null, text: null, retryNow: null };
  function showWake(seconds, attempt) {
    wake.banner = wake.banner || TL.$("#wake-banner");
    wake.text = wake.text || TL.$("#wake-text");
    if (!wake.banner) return;
    wake.banner.hidden = false;
    wake.text.textContent = "The free server is waking up or restarting (this can take up to a minute). Retrying in " + seconds + "s… (attempt " + attempt + ")";
  }
  function hideWake() { var b = TL.$("#wake-banner"); if (b) b.hidden = true; }
  var retryButton = TL.$("#wake-retry");
  if (retryButton) retryButton.addEventListener("click", function () { if (wake.retryNow) wake.retryNow(); });

  function delay(seconds, attempt) {
    return new Promise(function (resolve) {
      var left = seconds;
      showWake(left, attempt);
      var iv = setInterval(function () {
        left -= 1;
        if (left <= 0) { clearInterval(iv); wake.retryNow = null; resolve(); }
        else showWake(left, attempt);
      }, 1000);
      wake.retryNow = function () { clearInterval(iv); wake.retryNow = null; resolve(); };
    });
  }

  function ApiError(message, status, code, details) {
    this.message = message; this.status = status; this.code = code; this.details = details;
  }
  ApiError.prototype = Object.create(Error.prototype);
  TL.ApiError = ApiError;

  function parseError(status, body) {
    var err = body && body.error;
    if (err && typeof err === "object") return new ApiError(err.message || "Request failed.", status, err.code, err.details);
    if (typeof err === "string") return new ApiError(err, status, body.code, body.details);
    return new ApiError("Request failed (HTTP " + status + "). Please try again.", status, "http_error");
  }

  var RETRY_DELAYS = [3, 5, 8, 12, 15, 20];
  var RETRY_STATUS = { 502: true, 503: true, 504: true };

  /** fetch JSON with CSRF header and automatic retry while the free server wakes up. */
  TL.api = function (url, options) {
    options = options || {};
    var method = (options.method || "GET").toUpperCase();
    var headers = Object.assign({ Accept: "application/json" }, options.headers || {});
    if (method !== "GET") headers["X-CSRF-Token"] = TL.csrf();
    var body = options.body;
    if (options.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(options.json); }
    var attempt = 0;
    var retryable = options.retry !== false && (method === "GET" || options.retry === true);
    function run() {
      return fetch(url, { method: method, headers: headers, body: body, credentials: "same-origin", keepalive: !!options.keepalive })
        .then(function (response) {
          if (RETRY_STATUS[response.status] && retryable && attempt < RETRY_DELAYS.length) throw { retry: true };
          var type = response.headers.get("Content-Type") || "";
          var parse = type.indexOf("application/json") !== -1 ? response.json() : response.text().then(function () { return {}; });
          return parse.then(function (data) {
            hideWake();
            if (!response.ok) throw parseError(response.status, data);
            return data;
          });
        })
        .catch(function (error) {
          if (error instanceof ApiError) throw error;
          var networkFailure = error && (error.retry || error instanceof TypeError);
          if (networkFailure && retryable && attempt < RETRY_DELAYS.length) {
            var wait = RETRY_DELAYS[attempt];
            attempt += 1;
            return delay(wait, attempt).then(run);
          }
          hideWake();
          throw new ApiError("Could not reach the server. Check your connection and try again.", 0, "network_error");
        });
    }
    return run();
  };

  /* ---------------------------------------------------------------- theme toggle */
  function currentTheme() {
    var explicit = document.documentElement.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function syncThemeButton() {
    var button = TL.$("#theme-toggle");
    if (!button) return;
    var dark = currentTheme() === "dark";
    button.setAttribute("aria-pressed", dark ? "true" : "false");
    var label = dark ? "Switch to light theme" : "Switch to dark theme";
    TL.$("#theme-toggle-label").textContent = label;
    button.setAttribute("data-tip", label);
  }
  var toggle = TL.$("#theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      var html = document.documentElement;
      html.classList.add("theme-anim");
      html.setAttribute("data-theme", next);
      setTimeout(function () { html.classList.remove("theme-anim"); }, 320);
      try { localStorage.setItem("tl:theme", next); } catch (e) { /* ignore */ }
      syncThemeButton();
    });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (mq.addEventListener) mq.addEventListener("change", syncThemeButton);
    }
    syncThemeButton();
  }

  /* ---------------------------------------------------------------- CSV (client-side export of current view) */
  TL.csvCell = function (value) {
    var text = value === null || value === undefined ? "" : String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text; // spreadsheet formula injection guard
    if (/[",\n\r]/.test(text)) text = '"' + text.replace(/"/g, '""') + '"';
    return text;
  };
  TL.downloadText = function (filename, text, type) {
    var blob = new Blob(["﻿" + text], { type: type || "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = TL.el("a", { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(url); a.remove(); }, 0);
  };
})();
