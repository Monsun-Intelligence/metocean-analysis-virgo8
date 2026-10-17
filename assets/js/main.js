(() => {
  const PRODUCTS = [
    {
      id: "inacawo-current",
      label: "Arus permukaan (INACAWO 3 km)",
      dir: "figures/spatial/inacawo/current",
      stepHours: 1,
      end: "2026091623",
    },
    {
      id: "inacawo-wind",
      label: "Angin (INACAWO 3 km)",
      dir: "figures/spatial/inacawo/wind",
      stepHours: 1,
      end: "2026091623",
    },
    {
      id: "inacawo-swh",
      label: "Significant wave height (INACAWO 3 km)",
      dir: "figures/spatial/inacawo/swh",
      stepHours: 1,
      end: "2026091623",
    },
    {
      id: "inacawo-swell",
      label: "Swell height (INACAWO 3 km)",
      dir: "figures/spatial/inacawo/swell",
      stepHours: 1,
      end: "2026091623",
    },
    {
      id: "swh",
      label: "Significant wave height (GFS Wave)",
      dir: "figures/spatial/gfs/swh",
      stepHours: 3,
      end: "2026091521",
    },
    {
      id: "swell",
      label: "Primary swell (GFS Wave)",
      dir: "figures/spatial/gfs/swell",
      stepHours: 3,
      end: "2026091521",
    },
    {
      id: "wind10m",
      label: "Angin 10 m (GFS Atmos)",
      dir: "figures/spatial/gfs/wind10m",
      stepHours: 3,
      end: "2026091521",
    },
    {
      id: "seacurrent",
      label: "Arus permukaan (HYCOM)",
      dir: "figures/spatial/hycom/seacurrent",
      stepHours: 3,
      end: "2026091421", // NCSS incomplete for 15 Sep; covers event ± window core
    },
  ];

  const START_UTC = Date.UTC(2026, 8, 10, 0, 0, 0); // 2026-09-10
  const DEFAULT_STAMP = "2026091218";
  const PLAY_MS = 650;

  function parseStamp(stamp) {
    return Date.UTC(
      Number(stamp.slice(0, 4)),
      Number(stamp.slice(4, 6)) - 1,
      Number(stamp.slice(6, 8)),
      Number(stamp.slice(8, 10)),
      0,
      0
    );
  }

  function toStamp(ms) {
    const d = new Date(ms);
    const y = d.getUTCFullYear();
    const m = String(d.getUTCMonth() + 1).padStart(2, "0");
    const day = String(d.getUTCDate()).padStart(2, "0");
    const h = String(d.getUTCHours()).padStart(2, "0");
    return `${y}${m}${day}${h}`;
  }

  function buildStamps(product) {
    const out = [];
    const end = parseStamp(product.end || "2026091521");
    const step = (product.stepHours || 3) * 3600 * 1000;
    for (let t = START_UTC; t <= end; t += step) {
      out.push(toStamp(t));
    }
    return out;
  }

  function formatStamp(stamp) {
    const y = stamp.slice(0, 4);
    const m = stamp.slice(4, 6);
    const d = stamp.slice(6, 8);
    const h = stamp.slice(8, 10);
    return `${y}-${m}-${d} ${h}:00 UTC`;
  }

  function nearestIndex(stamps, target) {
    const want = parseStamp(target);
    let best = 0;
    let bestDiff = Infinity;
    stamps.forEach((s, i) => {
      const diff = Math.abs(parseStamp(s) - want);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = i;
      }
    });
    return best;
  }

  function initViewer() {
    const root = document.getElementById("spatial-viewer");
    if (!root) return;

    const productSelect = root.querySelector("[data-product]");
    const timeRange = root.querySelector("[data-time]");
    const readout = root.querySelector("[data-readout]");
    const img = root.querySelector("[data-frame]");
    const btnPrev = root.querySelector("[data-prev]");
    const btnNext = root.querySelector("[data-next]");
    const btnPlay = root.querySelector("[data-play]");

    let stamps = [];
    let playing = false;
    let timer = null;

    PRODUCTS.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label;
      productSelect.appendChild(opt);
    });

    function currentProduct() {
      return PRODUCTS.find((p) => p.id === productSelect.value) || PRODUCTS[0];
    }

    function setPlaying(on) {
      playing = on;
      if (btnPlay) {
        btnPlay.textContent = on ? "Pause" : "Play";
        btnPlay.setAttribute("aria-pressed", on ? "true" : "false");
      }
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      if (on) {
        timer = setInterval(() => {
          const max = Number(timeRange.max);
          let idx = Number(timeRange.value) + 1;
          if (idx > max) idx = 0;
          timeRange.value = String(idx);
          render();
        }, PLAY_MS);
      }
    }

    function syncSlider() {
      const product = currentProduct();
      stamps = buildStamps(product);
      const prevStamp =
        timeRange.dataset.stamp ||
        stamps[Number(timeRange.value)] ||
        DEFAULT_STAMP;
      timeRange.min = "0";
      timeRange.max = String(Math.max(stamps.length - 1, 0));
      timeRange.step = "1";
      timeRange.value = String(nearestIndex(stamps, prevStamp));
    }

    function step(delta) {
      setPlaying(false);
      const max = Number(timeRange.max);
      let idx = Number(timeRange.value) + delta;
      if (idx < 0) idx = max;
      if (idx > max) idx = 0;
      timeRange.value = String(idx);
      render();
    }

    function render() {
      const idx = Number(timeRange.value);
      const stamp = stamps[idx];
      if (!stamp) return;
      timeRange.dataset.stamp = stamp;
      const product = currentProduct();
      const src = `${product.dir}/t_${stamp}.png`;
      readout.textContent = formatStamp(stamp);
      img.src = src;
      img.alt = `${product.label} — ${formatStamp(stamp)}`;
      img.onerror = () => {
        img.removeAttribute("src");
        img.alt = "Frame tidak tersedia";
      };
    }

    productSelect.addEventListener("change", () => {
      setPlaying(false);
      syncSlider();
      render();
    });
    timeRange.addEventListener("input", () => {
      setPlaying(false);
      render();
    });
    btnPrev?.addEventListener("click", () => step(-1));
    btnNext?.addEventListener("click", () => step(1));
    btnPlay?.addEventListener("click", () => setPlaying(!playing));

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) setPlaying(false);
    });

    syncSlider();
    render();
  }

  document.addEventListener("DOMContentLoaded", () => {
    initViewer();
    initModelMenu();
  });

  function initModelMenu() {
    const root = document.querySelector("[data-model-menu]");
    if (!root) return;
    const tabs = [...root.querySelectorAll("[data-model-tab]")];
    const panels = [...root.querySelectorAll("[data-model-panel]")];

    function activate(id) {
      tabs.forEach((tab) => {
        const on = tab.getAttribute("data-model-tab") === id;
        tab.classList.toggle("is-active", on);
        tab.setAttribute("aria-selected", on ? "true" : "false");
      });
      panels.forEach((panel) => {
        const on = panel.getAttribute("data-model-panel") === id;
        panel.classList.toggle("is-active", on);
        if (on) panel.removeAttribute("hidden");
        else panel.setAttribute("hidden", "");
      });
    }

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        activate(tab.getAttribute("data-model-tab"));
      });
    });
  }
})();
