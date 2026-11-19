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

  TL.svg = function (path, label) {
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    var p = document.createElementNS(ns, "path");
    p.setAttribute("d", path);
    p.setAttribute("fill", "currentColor");
    svg.appendChild(p);
    return svg;
  };
  TL.icons = {
    star: "M12 17.3l-6.2 3.7 1.6-7L2 9.3l7.1-.6L12 2l2.9 6.7 7.1.6-5.4 4.7 1.6 7z",
    starOutline: "M12 15.4l-3.8 2.3 1-4.3-3.3-2.9 4.4-.4L12 6.1l1.7 4 4.4.4-3.3 2.9 1 4.3zM22 9.2l-7.2-.6L12 2 9.2 8.6 2 9.2l5.5 4.7L5.8 21 12 17.3 18.2 21l-1.6-7.1z",
    file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zm0 2.5L17.5 8H14zM8 13h8v2H8zm0 4h8v2H8z",
    close: "M18.3 5.7L12 12l6.3 6.3-1.4 1.4L10.6 13.4 4.3 19.7 2.9 18.3 9.2 12 2.9 5.7 4.3 4.3l6.3 6.3 6.3-6.3z",
    upload: "M5 20h14v-2H5zm7-16l-5 5h3v6h4V9h3z",
    inbox: "M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm0 12h-4a3 3 0 0 1-6 0H5V5h14z"
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
      [TL.el("span", { class: "toast-msg", text: message })]);
    var timer;
    function close() { clearTimeout(timer); if (toast.parentNode) toast.parentNode.removeChild(toast); if (opts.onClose) opts.onClose(); }
    if (opts.action) {
      toast.appendChild(TL.el("button", { type: "button", class: "btn btn-sm", text: opts.action.label, onclick: function () {
        opts.onClose = null; close(); opts.action.onClick();
      } }));
    }
    toast.appendChild(TL.el("button", { type: "button", class: "btn btn-sm", "aria-label": "Dismiss notification", text: "×", onclick: close }));
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
    button.title = label;
  }
  var toggle = TL.$("#theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
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
