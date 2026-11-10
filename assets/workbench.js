(() => {
  // Only theme and viewport affordances; Gradio handles native page navigation.
  const key = "h3-studio-preview-theme";
  const root = document.documentElement;
  let theme = "dark";
  try {
    const saved = localStorage.getItem(key);
    if (saved === "light" || saved === "dark") theme = saved;
  } catch (_) {}

  function apply(next) {
    root.setAttribute("data-h3-theme", next);
    document.getElementById("h3-root")?.setAttribute("data-h3-theme", next);
    root.style.colorScheme = next;
    const button = document.getElementById("h3-theme-toggle");
    if (button) {
      button.setAttribute("title", next === "dark" ? "切换到日间主题" : "切换到夜间主题");
      button.setAttribute("aria-pressed", next === "light" ? "true" : "false");
    }
  }
  apply(theme);
  const observer = new MutationObserver(() => {
    const app = document.getElementById("h3-root");
    if (app) {
      app.setAttribute("data-h3-theme", theme);
      observer.disconnect();
    }
  });
  observer.observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.("#h3-theme-toggle");
    if (button) {
      theme = root.getAttribute("data-h3-theme") === "dark" ? "light" : "dark";
      apply(theme);
      try { localStorage.setItem(key, theme); } catch (_) {}
    }
    const nav = event.target.closest?.(".h3-nav");
    if (nav) {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, true);
})();
