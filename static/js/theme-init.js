/* Runs before first paint to apply the saved theme and avoid a flash. */
(function () {
  try {
    var saved = localStorage.getItem("tl:theme");
    if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* storage unavailable: follow prefers-color-scheme */ }
})();
