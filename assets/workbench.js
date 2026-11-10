() => {
  // Browser-only theme control and presentational motion layer.
  // No Gradio component values, API calls, or generation events are read,
  // changed, or submitted by this script.
  const storageKey = "nm-videomodel-workbench-theme";
  const root = document.documentElement;
  let selected = "dark";
  try {
    const saved = window.localStorage.getItem(storageKey);
    if (saved === "light" || saved === "dark") selected = saved;
  } catch (_error) {}

  function applyTheme(mode) {
    root.setAttribute("data-wb-theme", mode);
    root.style.colorScheme = mode;
    const button = document.getElementById("wb-theme-toggle");
    if (button) {
      button.setAttribute("aria-pressed", String(mode === "light"));
      button.setAttribute("title", mode === "light" ? "切换到夜间模式" : "切换到日间模式");
    }
  }

  function markReady() {
    const app = document.getElementById("wb-root");
    if (app) app.classList.add("wb-ready");
  }

  function bindMotion() {
    const app = document.getElementById("wb-root");
    if (!app || app.dataset.motionBound === "1") return;
    app.dataset.motionBound = "1";
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) {
      app.addEventListener('pointermove', (event) => {
        const rect = app.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * 100;
        const y = ((event.clientY - rect.top) / rect.height) * 100;
        app.style.setProperty('--wb-mx', x.toFixed(2) + '%');
        app.style.setProperty('--wb-my', y.toFixed(2) + '%');
      }, { passive: true });
    }
    app.querySelectorAll('.wb-side-link').forEach((link) => {
      link.addEventListener('click', () => {
        app.querySelectorAll('.wb-side-link.is-active').forEach((item) => item.classList.remove('is-active'));
        link.classList.add('is-active');
      });
    });
  }

  function init() {
    applyTheme(selected);
    bindMotion();
    requestAnimationFrame(markReady);
  }

  document.addEventListener("click", (event) => {
    const element = event.target;
    const toggle = element && element.closest && element.closest("#wb-theme-toggle");
    if (!toggle) return;
    const next = root.getAttribute("data-wb-theme") === "light" ? "dark" : "light";
    applyTheme(next);
    try {
      window.localStorage.setItem(storageKey, next);
    } catch (_error) {}
  });


  // Cosmetic waiting layer. We do not wrap, prevent, call or modify the
  // original Gradio run_btn.click handler, its parameters, queue or API.
  const waitState = {
    active:false, started:0, initialMedia:'', wasBusy:false,
    interval:null, errorBaseline:[]
  };
  const $ = (selector) => document.querySelector(selector);
  function currentMediaSignature() {
    const host = $('#wb-output-video');
    if (!host) return '';
    const parts = [];
    host.querySelectorAll('video, source, a[download], a[href$=".mp4"]').forEach((item) => {
      const url = item.currentSrc || item.getAttribute('src') || item.getAttribute('href') || '';
      if (url && url !== 'about:blank') parts.push(url);
    });
    return parts.join('|');
  }
  function elapsedSeconds() { return Math.max(0, Math.floor((Date.now() - waitState.started) / 1000)); }
  function updateClock() {
    const clock = $('#wb-wait-clock');
    if (!clock || !waitState.active) return;
    const n = elapsedSeconds();
    const mins = Math.floor(n / 60);
    const secs = n % 60;
    clock.textContent = String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  }
  function visibleErrors() {
    const nodes = document.querySelectorAll('.toast, [role="alert"], .error, .toast-wrap');
    const result = [];
    for (const node of nodes) {
      if (node.closest('#wb-wait-layer')) continue;
      const str = String(node.textContent || '').trim().slice(0, 280);
      if (str && /error|错误|失败|异常|invalid|traceback|exception/i.test(str)) result.push(str);
    }
    return result;
  }
  function generateButton() {
    const rootButton = $('#wb-generate');
    if (!rootButton) return null;
    return rootButton.matches('button') ? rootButton : rootButton.querySelector('button');
  }
  function buttonBusy() {
    const btn = generateButton();
    return !!(btn && (btn.disabled || btn.getAttribute('aria-busy') === 'true' || btn.classList.contains('loading')));
  }
  function stopWaiting() {
    if (!waitState.active) return;
    waitState.active = false;
    if (waitState.interval !== null) window.clearInterval(waitState.interval);
    waitState.interval = null;
    const panel = $('#wb-root');
    if (panel) {
      panel.classList.remove('is-generating');
      panel.classList.remove('wb-minimal-effects');
    }
    const effectToggle = $('#wb-effect-toggle');
    if (effectToggle) {
      effectToggle.setAttribute('aria-pressed', 'false');
      effectToggle.textContent = '关闭动态光效';
    }
  }
  function inspectWait() {
    if (!waitState.active) return;
    updateClock();
    const elapsed = elapsedSeconds();
    const media = currentMediaSignature();
    if (elapsed >= 1 && media && media !== waitState.initialMedia) {
      stopWaiting();
      return;
    }
    const busy = buttonBusy();
    if (busy) waitState.wasBusy = true;
    if (elapsed >= 2 && waitState.wasBusy && !busy) {
      stopWaiting();
      return;
    }
    if (elapsed >= 1 && visibleErrors().some(msg => !waitState.errorBaseline.includes(msg))) {
      stopWaiting();
    }
  }
  function startWaiting() {
    stopWaiting();
    const panel = $('#wb-root');
    const layer = $('#wb-wait-layer');
    if (!panel || !layer) return;
    waitState.active = true;
    waitState.started = Date.now();
    waitState.initialMedia = currentMediaSignature();
    waitState.wasBusy = buttonBusy();
    waitState.errorBaseline = visibleErrors();
    panel.classList.add('is-generating');
    const msg = $('#wb-wait-desc');
    if (msg) msg.textContent = '视频任务已发起，正在等待推理服务返回结果';
    updateClock();
    waitState.interval = window.setInterval(inspectWait, 450);
  }
  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!target || !target.closest) return;
    if (target.closest('#wb-effect-toggle')) {
      const panel = $('#wb-root');
      if (!panel) return;
      const minimal = panel.classList.toggle('wb-minimal-effects');
      const button = $('#wb-effect-toggle');
      if (button) {
        button.setAttribute('aria-pressed', String(minimal));
        button.textContent = minimal ? '开启动态光效' : '关闭动态光效';
      }
      return;
    }
    if (target.closest('#wb-wait-dismiss')) {
      // Presentation-only dismissal; original generation keeps running.
      stopWaiting();
      return;
    }
  });
  // Capturing the original click happens before Gradio can immediately disable
  // the submit button. The click is never prevented or altered.
  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!target || !target.closest || !target.closest('#wb-generate')) return;
    const btn = generateButton();
    if (btn && !btn.disabled && btn.getAttribute('aria-disabled') !== 'true') {
      startWaiting();
    }
  }, true);
  document.addEventListener('loadedmetadata', (event) => {
    if (!waitState.active) return;
    const video = event.target;
    if (video && video.matches && video.matches('#wb-output-video video')) inspectWait();
  }, true);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
}