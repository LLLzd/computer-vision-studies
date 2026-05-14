const state = {
  meta: null,
  scenes: [],
  l1: "raw",
  channel: "CAM_FRONT",
  searchPage: 1,
  searchTotal: 0,
  frames: [],
  frameIdx: 0,
  playTimer: null,
  selectedSample: null,
};

const $ = (id) => document.getElementById(id);

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
    CAM_FRONT_LEFT: "左前相机",
    CAM_FRONT: "前相机",
    CAM_FRONT_RIGHT: "右前相机",
    CAM_BACK_LEFT: "左后相机",
    CAM_BACK: "后相机",
    CAM_BACK_RIGHT: "右后相机",
    LIDAR_TOP: "顶部 LiDAR",
  };
  return map[ch] || ch;
}

function mkL2Btn(value, text, forceL1) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "l2";
  b.dataset.channel = value;
  b.textContent = text;
  b.addEventListener("click", () => {
    document.querySelectorAll(".nav-l2 .l2").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.l1 = forceL1 || "raw";
    state.channel = value;
    syncL1UI();
    $("fig-cam2d-wrap").classList.toggle("hidden", !String(value).startsWith("2d:"));
    refreshSampleView();
  });
  return b;
}

function buildNavL2() {
  const raw = $("nav-l2-raw");
  const label = $("nav-l2-label");
  raw.innerHTML = "";
  label.innerHTML = "";
  if (!state.meta) return;

  raw.appendChild(
    Object.assign(document.createElement("p"), { className: "hint", textContent: "传感器 / 数据类型" }),
  );
  for (const ch of state.meta.camera_channels) {
    raw.appendChild(mkL2Btn(ch, `${channelLabel(ch)} (${ch})`));
  }
  raw.appendChild(mkL2Btn("LIDAR_TOP", `${channelLabel("LIDAR_TOP")} (BEV 渲染)`));
  for (const ch of state.meta.radar_channels) {
    raw.appendChild(mkL2Btn(ch, ch));
  }

  label.appendChild(
    Object.assign(document.createElement("p"), { className: "hint", textContent: "标注 / 定位" }),
  );
  for (const ch of state.meta.camera_channels) {
    label.appendChild(mkL2Btn(`2d:${ch}`, `2D 投影 (${channelLabel(ch)})`, "label"));
  }
  label.appendChild(mkL2Btn("3d:bev", "3D 物体 BEV", "label"));
  label.appendChild(mkL2Btn("loc:ego", "Localization (ego)", "label"));
}

function syncL1UI() {
  document.querySelectorAll(".nav-l1 .l1").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.l1 === state.l1);
  });
  $("nav-l2-raw").classList.toggle("hidden", state.l1 !== "raw");
  $("nav-l2-label").classList.toggle("hidden", state.l1 !== "label");
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
  buildNavL2();
  const firstCam = state.meta.camera_channels[0] || "CAM_FRONT";
  state.channel = firstCam;
  document.querySelector(`#nav-l2-raw .l2[data-channel="${firstCam}"]`)?.classList.add("active");
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

async function selectSample(sampleToken, tr) {
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
    $("sample-summary").textContent = "在结果中点击一条样本；左侧选择通道。";
    return;
  }
  const st = state.selectedSample;
  const detail = await api(`/api/samples/${st}`);
  $("sample-summary").textContent = `${detail.scene_name} · ${detail.data_channels.length} 通道 · ${detail.ann_count} 个标注`;

  const ego = await api(`/api/samples/${st}/ego`);
  $("ego-json").textContent = JSON.stringify(ego, null, 2);

  const ch = state.channel;
  const bust = `&t=${Date.now()}`;

  $("img-lidar").src = `/api/render/lidar_bev?sample_token=${encodeURIComponent(st)}${bust}`;
  $("img-annbev").src = `/api/render/ann_bev?sample_token=${encodeURIComponent(st)}${bust}`;

  if (String(ch).startsWith("2d:")) {
    const cam = ch.slice(3);
    const url = `/api/render/camera_2d?sample_token=${encodeURIComponent(st)}&channel=${encodeURIComponent(cam)}${bust}`;
    $("frame-img").src = url;
    $("img-cam2d").src = url;
    $("fig-cam2d-wrap").classList.remove("hidden");
    $("frame-label").textContent = `2D 投影 · ${cam}`;
  } else {
    $("fig-cam2d-wrap").classList.add("hidden");

    if (ch === "3d:bev") {
      $("frame-img").src = `/api/render/ann_bev?sample_token=${encodeURIComponent(st)}${bust}`;
      $("frame-label").textContent = "3D 标注 BEV";
    } else if (ch === "loc:ego") {
      $("frame-img").removeAttribute("src");
      $("frame-label").textContent = "定位数据见右侧 ego_pose JSON";
    } else if (ch === "LIDAR_TOP") {
      $("frame-img").src = `/api/render/lidar_bev?sample_token=${encodeURIComponent(st)}${bust}`;
      $("frame-label").textContent = "LiDAR 点云 BEV（服务端渲染）";
    } else if (!detail.data_channels.includes(ch)) {
      $("frame-img").removeAttribute("src");
      $("frame-label").textContent = `当前样本无通道 ${ch}`;
    } else {
      const m = await api(`/api/samples/${st}/media?channel=${encodeURIComponent(ch)}`);
      $("frame-img").src = m.media_url;
      $("frame-label").textContent = `${ch} · 原始文件`;
    }
  }

  await loadAnnSummary(st);
}

async function loadClip() {
  stopPlay();
  const scene = currentSceneToken();
  let ch = state.channel;
  if (String(ch).startsWith("2d:")) ch = ch.slice(3);
  if (ch === "3d:bev" || ch === "loc:ego") ch = "CAM_FRONT";
  if (ch === "LIDAR_TOP") ch = "CAM_FRONT";
  const step = $("clip-step").value;
  const maxf = $("clip-max").value;
  const data = await api(
    `/api/clips/frames?scene_token=${encodeURIComponent(scene)}&channel=${encodeURIComponent(ch)}&step=${step}&max_frames=${maxf}`,
  );
  state.frames = data.frames;
  state.frameIdx = 0;
  $("btn-play").disabled = state.frames.length === 0;
  $("btn-stop").disabled = true;
  showFrame(0);
}

function showFrame(i) {
  if (!state.frames.length) return;
  state.frameIdx = ((i % state.frames.length) + state.frames.length) % state.frames.length;
  const f = state.frames[state.frameIdx];
  $("frame-img").src = f.media_url;
  $("frame-label").textContent = `${f.filename} (#${state.frameIdx + 1}/${state.frames.length})`;
}

function stopPlay() {
  if (state.playTimer) clearInterval(state.playTimer);
  state.playTimer = null;
  $("btn-stop").disabled = true;
}

function startPlay() {
  if (!state.frames.length) return;
  stopPlay();
  $("btn-stop").disabled = false;
  state.playTimer = setInterval(() => showFrame(state.frameIdx + 1), 380);
}

function initEvents() {
  document.querySelectorAll(".nav-l1 .l1").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.l1 = btn.dataset.l1;
      if (state.l1 === "raw" && String(state.channel).startsWith("2d:")) {
        state.channel = state.meta?.camera_channels?.[0] || "CAM_FRONT";
        document.querySelectorAll(".nav-l2 .l2").forEach((x) => x.classList.remove("active"));
        document.querySelector(`#nav-l2-raw .l2[data-channel="${state.channel}"]`)?.classList.add("active");
      }
      if (state.l1 === "label" && !String(state.channel).startsWith("2d:") && state.channel !== "3d:bev" && state.channel !== "loc:ego") {
        state.channel = `2d:${state.meta?.camera_channels?.[0] || "CAM_FRONT"}`;
        document.querySelectorAll(".nav-l2 .l2").forEach((x) => x.classList.remove("active"));
        document.querySelector(`#nav-l2-label .l2[data-channel="${state.channel}"]`)?.classList.add("active");
      }
      syncL1UI();
      $("fig-cam2d-wrap").classList.toggle("hidden", !String(state.channel).startsWith("2d:"));
      refreshSampleView();
    });
  });
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
}

async function boot() {
  initEvents();
  await loadMetaAndScenes();
  syncL1UI();
  await runSearch(1);
  const first = $("results-body").querySelector("tr");
  if (first?.dataset.sample) await selectSample(first.dataset.sample, first);
}

boot().catch((e) => {
  $("sample-summary").textContent = `启动失败: ${e.message}`;
  console.error(e);
});
