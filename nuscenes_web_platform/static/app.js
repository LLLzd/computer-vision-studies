const state = {
  meta: null,
  scenes: [],
  searchPage: 1,
  searchTotal: 0,
  frames: [],
  frameIdx: 0,
  playTimer: null,
  selectedSample: null,
  cachedDetail: null,
  egoTrail: null,
};

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status}: ${t}`);
  }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) return r.json();
  return r.text();
}

function channelLabel(ch) {
  const map = {
    CAM_FRONT_LEFT: "左前",
    CAM_FRONT: "前",
    CAM_FRONT_RIGHT: "右前",
    CAM_BACK_LEFT: "左后",
    CAM_BACK: "后",
    CAM_BACK_RIGHT: "右后",
    LIDAR_TOP: "LiDAR",
  };
  return map[ch] || ch;
}

function isPlaying() {
  return state.playTimer != null;
}

function readLayerSettings() {
  const cams = {};
  for (const ch of state.meta?.camera_channels || []) {
    const cb = document.querySelector(`#chk-cam-${ch}`);
    cams[ch] = cb ? cb.checked : false;
  }
  return {
    cameras: cams,
    lidar: $("chk-lidar")?.checked ?? false,
    gtLidarBoxes: $("chk-gt-lidar")?.checked ?? false,
    gtCamProj: $("chk-gt-cam")?.checked ?? false,
  };
}

function cacheBustQuery() {
  const i = isPlaying() ? state.frameIdx : 0;
  return `_=${Date.now()}&i=${i}`;
}

function applyVisuals(sampleToken) {
  if (!sampleToken || !state.meta) return;
  const L = readLayerSettings();
  const q = cacheBustQuery();
  const detail = state.cachedDetail;

  for (const ch of state.meta.camera_channels) {
    const img = document.querySelector(`#cam-img-${ch}`);
    const slot = img?.closest(".cam-slot");
    if (!img || !slot) continue;

    if (!L.cameras[ch]) {
      img.removeAttribute("src");
      slot.classList.add("inactive");
      continue;
    }
    if (!isPlaying() && detail && !detail.data_channels.includes(ch)) {
      img.removeAttribute("src");
      slot.classList.add("inactive");
      continue;
    }

    slot.classList.remove("inactive");
    if (L.gtCamProj) {
      img.src = `/api/render/camera_2d?sample_token=${encodeURIComponent(sampleToken)}&channel=${encodeURIComponent(ch)}&boxes=true&${q}`;
    } else {
      img.src = `/api/samples/${encodeURIComponent(sampleToken)}/raw_image/${encodeURIComponent(ch)}?${q}`;
    }
  }

  const lidarImg = $("img-lidar");
  const lidarSlot = $("lidar-slot");
  if (!L.lidar) {
    lidarImg.removeAttribute("src");
    lidarSlot.classList.add("inactive");
  } else {
    lidarSlot.classList.remove("inactive");
    const boxes = L.gtLidarBoxes ? "true" : "false";
    lidarImg.src = `/api/render/lidar_bev?sample_token=${encodeURIComponent(sampleToken)}&boxes=${boxes}&${q}`;
  }
}

function buildCamGridAndChecks() {
  const grid = $("cam-grid");
  const chkCams = $("chk-cams");
  grid.innerHTML = "";
  chkCams.innerHTML = "";
  if (!state.meta) return;

  for (const ch of state.meta.camera_channels) {
    const slot = document.createElement("div");
    slot.className = "cam-slot";
    slot.dataset.channel = ch;
    const cap = document.createElement("div");
    cap.className = "cam-cap";
    cap.textContent = `${channelLabel(ch)} · ${ch}`;
    const img = document.createElement("img");
    img.alt = ch;
    img.id = `cam-img-${ch}`;
    slot.appendChild(cap);
    slot.appendChild(img);
    grid.appendChild(slot);

    const lab = document.createElement("label");
    lab.className = "chk-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = `chk-cam-${ch}`;
    cb.checked = true;
    cb.addEventListener("change", () => {
      if (state.selectedSample) applyVisuals(state.selectedSample);
    });
    lab.appendChild(cb);
    const span = document.createElement("span");
    span.textContent = `${channelLabel(ch)} (${ch})`;
    lab.appendChild(span);
    chkCams.appendChild(lab);
  }
}

function wireGtAndLidarChecks() {
  ["chk-lidar", "chk-gt-lidar", "chk-gt-cam"].forEach((id) => {
    $(id)?.addEventListener("change", () => {
      if (state.selectedSample) applyVisuals(state.selectedSample);
    });
  });
}

async function loadMetaAndScenes() {
  state.meta = await api("/api/meta");
  const dl = $("cat-list");
  dl.innerHTML = "";
  for (const c of state.meta.categories.slice(0, 200)) {
    const o = document.createElement("option");
    o.value = c;
    dl.appendChild(o);
  }
  state.scenes = await api("/api/scenes");
  const sel = $("scene-select");
  sel.innerHTML = "";
  for (const s of state.scenes) {
    const o = document.createElement("option");
    o.value = s.token;
    o.textContent = `${s.name} (${s.nbr_samples})`;
    sel.appendChild(o);
  }
  buildCamGridAndChecks();
}

async function fetchEgoTrail() {
  const sc = currentSceneToken();
  if (!sc) return;
  try {
    state.egoTrail = await api(`/api/scenes/${encodeURIComponent(sc)}/ego_trail`);
  } catch {
    state.egoTrail = null;
  }
  renderEgoTrailSvg();
  if (state.selectedSample) updateTrailMarker(state.selectedSample);
}

function renderEgoTrailSvg() {
  const svg = $("ego-trail-svg");
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const pts = state.egoTrail?.points;
  if (!pts?.length) {
    svg.removeAttribute("viewBox");
    return;
  }
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const pad = span * 0.08 + 2;
  const minx = Math.min(...xs) - pad;
  const maxx = Math.max(...xs) + pad;
  const miny = Math.min(...ys) - pad;
  const maxy = Math.max(...ys) + pad;
  const w = Math.max(maxx - minx, 1e-3);
  const h = Math.max(maxy - miny, 1e-3);
  svg.setAttribute("viewBox", `${minx} ${miny} ${w} ${h}`);

  const poly = document.createElementNS(SVG_NS, "polyline");
  poly.setAttribute("class", "trail-line");
  poly.setAttribute("points", pts.map((p) => `${p.x},${p.y}`).join(" "));
  svg.appendChild(poly);

  const c = document.createElementNS(SVG_NS, "circle");
  c.setAttribute("class", "trail-cursor");
  c.setAttribute("r", String(Math.max(Math.min(w, h) * 0.025, 0.35)));
  c.setAttribute("visibility", "hidden");
  svg.appendChild(c);
}

function updateTrailMarker(sampleToken) {
  const svg = $("ego-trail-svg");
  const cur = svg?.querySelector(".trail-cursor");
  if (!cur || !state.egoTrail?.points) return;
  const pt = state.egoTrail.points.find((p) => p.sample_token === sampleToken);
  if (!pt) {
    cur.setAttribute("visibility", "hidden");
    return;
  }
  cur.setAttribute("visibility", "visible");
  cur.setAttribute("cx", String(pt.x));
  cur.setAttribute("cy", String(pt.y));
}

async function refreshSidePanel(sampleToken) {
  if (!sampleToken) return;
  try {
    const ego = await api(`/api/samples/${sampleToken}/ego`);
    $("ego-json").textContent = JSON.stringify(ego, null, 2);
  } catch (e) {
    $("ego-json").textContent = `ego 加载失败: ${e.message}`;
  }
  updateTrailMarker(sampleToken);
}

function currentSceneToken() {
  return $("scene-select").value;
}

async function runSearch(page) {
  state.searchPage = page;
  const category = $("category-input").value.trim();
  const scene = currentSceneToken();
  const qs = new URLSearchParams({ page: String(page), page_size: "20" });
  if (category) qs.set("category", category);
  if (scene) qs.set("scene_token", scene);
  const data = await api(`/api/search?${qs}`);
  state.searchTotal = data.total;
  $("results-meta").textContent = `共 ${data.total} 条`;
  $("page-label").textContent = `第 ${data.page} 页`;
  const tb = $("results-body");
  tb.innerHTML = "";
  for (const row of data.items) {
    const tr = document.createElement("tr");
    tr.dataset.sample = row.sample_token;
    tr.innerHTML = `<td>${escapeHtml(row.scene_name)}</td><td>${row.timestamp}</td><td class="mono">${row.sample_token.slice(0, 8)}…</td>`;
    tr.addEventListener("click", () => selectSample(row.sample_token, tr));
    tb.appendChild(tr);
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function stopPlay() {
  if (state.playTimer) clearInterval(state.playTimer);
  state.playTimer = null;
  $("btn-stop").disabled = true;
  $("btn-play").disabled = state.frames.length === 0;
  $("playback-bar").classList.add("hidden");
}

async function selectSample(sampleToken, tr) {
  stopPlay();
  state.selectedSample = sampleToken;
  document.querySelectorAll("#results-body tr").forEach((r) => r.classList.remove("selected"));
  if (tr) tr.classList.add("selected");
  await refreshSampleView();
}

async function loadAnnSummary(st) {
  try {
    const ann = await api(`/api/samples/${st}/annotations`);
    $("ann-short").textContent = `${ann.items.length} 个实例 · 示例: ${ann.items
      .slice(0, 6)
      .map((a) => a.category_name)
      .join(", ")}`;
  } catch (e) {
    $("ann-short").textContent = `标注摘要加载失败: ${e.message}`;
  }
}

async function refreshSampleView() {
  if (!state.selectedSample) {
    $("sample-summary").textContent = "在结果中点击一条样本；左侧勾选要显示的传感器与真值。";
    return;
  }
  const st = state.selectedSample;
  const detail = await api(`/api/samples/${st}`);
  state.cachedDetail = detail;
  $("sample-summary").textContent = `${detail.scene_name} · ${detail.data_channels.length} 通道 · ${detail.ann_count} 个标注`;

  applyVisuals(st);
  await refreshSidePanel(st);
  await loadAnnSummary(st);
}

async function loadClip() {
  stopPlay();
  const scene = currentSceneToken();
  const step = $("clip-step").value;
  const maxf = $("clip-max").value;
  const data = await api(
    `/api/clips/frames?scene_token=${encodeURIComponent(scene)}&channel=${encodeURIComponent("CAM_FRONT")}&step=${step}&max_frames=${maxf}`,
  );
  state.frames = data.frames;
  state.frameIdx = 0;
  $("btn-play").disabled = state.frames.length === 0;
  $("btn-stop").disabled = true;
  if (state.frames.length) {
    try {
      state.cachedDetail = await api(`/api/samples/${state.frames[0].sample_token}`);
    } catch {
      /* keep previous cachedDetail */
    }
  }
  if (state.frames.length) {
    showFrame(0);
  }
}

function showFrame(i) {
  if (!state.frames.length) return;
  state.frameIdx = ((i % state.frames.length) + state.frames.length) % state.frames.length;
  const f = state.frames[state.frameIdx];
  applyVisuals(f.sample_token);
  void refreshSidePanel(f.sample_token);
  const bar = $("playback-bar");
  bar.classList.remove("hidden");
  bar.textContent = `播放中 · 第 ${state.frameIdx + 1}/${state.frames.length} 帧 · ${f.filename}`;
}

function startPlay() {
  if (!state.frames.length) return;
  stopPlay();
  $("btn-stop").disabled = false;
  state.playTimer = setInterval(() => showFrame(state.frameIdx + 1), 420);
}

function initEvents() {
  $("btn-search").addEventListener("click", () => runSearch(1));
  $("page-prev").addEventListener("click", () => {
    if (state.searchPage > 1) runSearch(state.searchPage - 1);
  });
  $("page-next").addEventListener("click", () => {
    const maxPage = Math.ceil(state.searchTotal / 20) || 1;
    if (state.searchPage < maxPage) runSearch(state.searchPage + 1);
  });
  $("btn-load-clip").addEventListener("click", () => loadClip());
  $("btn-play").addEventListener("click", () => startPlay());
  $("btn-stop").addEventListener("click", () => stopPlay());
  $("scene-select").addEventListener("change", () => {
    void fetchEgoTrail();
  });
}

async function boot() {
  initEvents();
  wireGtAndLidarChecks();
  await loadMetaAndScenes();
  await fetchEgoTrail();
  await runSearch(1);
  const first = $("results-body").querySelector("tr");
  if (first?.dataset.sample) await selectSample(first.dataset.sample, first);
}

boot().catch((e) => {
  $("sample-summary").textContent = `启动失败: ${e.message}`;
  console.error(e);
});
