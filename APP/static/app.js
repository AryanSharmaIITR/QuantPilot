"use strict";

const api = {
  async get(url) { return this._req(url); },
  async post(url, body) { return this._req(url, "POST", body); },
  async del(url) { return this._req(url, "DELETE"); },
  async _req(url, method = "GET", body) {
    const opts = { method, headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const res = await fetch(url, opts);
    const data = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
    if (!res.ok) throw new Error((data && data.detail) || res.statusText);
    return data;
  },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const fmtTime = (ts) => ts ? new Date(ts * 1000).toLocaleTimeString() : "—";

let selectedJob = null;
let pollTimer = null;

// ---------------- Tabs ----------------
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.remove("active"));
  $$(".panel").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  $("#" + t.dataset.tab).classList.add("active");
  if (t.dataset.tab === "dashboard") loadStatus();
  if (t.dataset.tab === "stocks") loadStocks();
  if (t.dataset.tab === "pipeline") loadJobs();
  if (t.dataset.tab === "predictions") loadPredictions();
}));

// ---------------- Dashboard ----------------
async function loadStatus() {
  try {
    const s = await api.get("/api/status");
    setBusy(s.busy);
    const card = (label, value, sub, cls = "") =>
      `<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub || ""}</div></div>`;
    $("#status-cards").innerHTML =
      card("Project", s.project, "active configuration") +
      card("Stocks", s.num_stocks, "in universe") +
      card("Market indices", s.num_market, "in universe") +
      card("Checkpoint", s.checkpoint_exists ? "Ready" : "Missing", s.checkpoint.split("/").pop(),
           s.checkpoint_exists ? "ok" : "bad") +
      card("Predictions", s.predictions_exist ? s.predictions_count : "None",
           s.predictions_as_of ? "as of " + s.predictions_as_of : "run predict pipeline",
           s.predictions_exist ? "ok" : "") +
      card("Pipeline", s.busy ? "Running" : "Idle", s.busy ? "a job is active" : "ready",
           s.busy ? "" : "ok");
  } catch (e) { $("#status-cards").innerHTML = `<div class="card bad">${e.message}</div>`; }
}

function setBusy(busy) {
  const pill = $("#busy-pill");
  pill.textContent = busy ? "● running" : "● idle";
  pill.className = "pill " + (busy ? "pill-busy" : "pill-idle");
}

// ---------------- Stocks ----------------
async function loadStocks() {
  const data = await api.get("/api/stocks");
  renderInstruments("stocks", data.stocks);
  renderInstruments("market", data.market);
  $("#stocks-count").textContent = data.stocks.length;
  $("#market-count").textContent = data.market.length;
}

function renderInstruments(kind, items) {
  $(`#${kind}-tbody`).innerHTML = items.map((it) => `
    <tr>
      <td>${it.category}</td>
      <td>${escapeHtml(it.name)}</td>
      <td class="mono">${escapeHtml(it.ticker)}</td>
      <td><button class="row-del" title="Remove" data-kind="${kind}" data-name="${escapeHtml(it.name)}">✕</button></td>
    </tr>`).join("");
  $$(`#${kind}-tbody .row-del`).forEach((b) => b.addEventListener("click", async () => {
    if (!confirm(`Remove ${b.dataset.name}?`)) return;
    try { await api.del(`/api/stocks/${b.dataset.kind}/${encodeURIComponent(b.dataset.name)}`); loadStocks(); }
    catch (e) { showError(e.message); }
  }));
}

$("#add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  try {
    await api.post("/api/stocks", {
      kind: $("#add-kind").value,
      name: $("#add-name").value,
      ticker: $("#add-ticker").value,
    });
    $("#add-name").value = ""; $("#add-ticker").value = "";
    loadStocks();
  } catch (e) { showError(e.message); }
});

$("#reset-btn").addEventListener("click", async () => {
  if (!confirm("Reset the universe to the built-in defaults?")) return;
  try { await api.post("/api/stocks/reset"); loadStocks(); } catch (e) { showError(e.message); }
});

const showError = (msg) => { $("#add-error").textContent = msg; };
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------- Pipeline ----------------
$$("[data-stage]").forEach((b) => b.addEventListener("click", () => runStage(b.dataset.stage, b.dataset.mode)));

async function runStage(stage, mode) {
  try {
    const job = await api.post("/api/pipeline/run", { stage, mode: mode || null });
    selectedJob = job.id;
    await loadJobs();
    startPolling();
  } catch (e) { alert(e.message); }
}

async function loadJobs() {
  const jobs = await api.get("/api/pipeline/jobs");
  setBusy(jobs.some((j) => j.status === "running"));
  $("#jobs-tbody").innerHTML = jobs.map((j) => `
    <tr class="clickable ${j.id === selectedJob ? "sel" : ""}" data-id="${j.id}">
      <td>${j.stage}${j.mode ? " <span class='muted'>(" + j.mode + ")</span>" : ""}</td>
      <td><span class="status-badge s-${j.status}">${j.status}</span></td>
      <td>${fmtTime(j.started)}</td>
      <td>${j.status === "running" ? `<button class="row-del stop-btn" data-id="${j.id}" title="Stop">■</button>` : ""}</td>
    </tr>`).join("") || `<tr><td colspan="4" class="muted">No jobs yet.</td></tr>`;

  $$("#jobs-tbody tr.clickable").forEach((tr) => tr.addEventListener("click", (e) => {
    if (e.target.classList.contains("stop-btn")) return;
    selectedJob = tr.dataset.id; loadJobs(); loadLog();
  }));
  $$("#jobs-tbody .stop-btn").forEach((b) => b.addEventListener("click", async (e) => {
    e.stopPropagation();
    await api.post(`/api/pipeline/jobs/${b.dataset.id}/stop`); loadJobs();
  }));
  if (selectedJob) loadLog();
}

async function loadLog() {
  if (!selectedJob) return;
  try {
    const res = await api.get(`/api/pipeline/jobs/${selectedJob}/log`);
    $("#log-job-id").textContent = "· " + selectedJob;
    const box = $("#job-log");
    box.textContent = res.log || "(no output yet)";
    box.scrollTop = box.scrollHeight;
  } catch (e) { /* job may not exist */ }
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    const jobs = await api.get("/api/pipeline/jobs");
    await loadJobs();
    if (!jobs.some((j) => j.status === "running")) { clearInterval(pollTimer); pollTimer = null; }
  }, 2000);
}

// ---------------- Predictions ----------------
$("#refresh-pred").addEventListener("click", loadPredictions);
$("#run-pred").addEventListener("click", () => runStage("predict-pipeline"));

async function loadPredictions() {
  const data = await api.get("/api/predictions");
  $("#pred-meta").textContent = data.count
    ? `${data.count} signals · as of ${data.predictions[0].as_of_date}`
    : "No predictions yet — run the predict pipeline.";
  $("#pred-tbody").innerHTML = data.predictions.map((p, i) => {
    const prob = parseFloat(p.up_probability);
    const up = (p.signal || "").toUpperCase() === "UP";
    const pct = Math.round(prob * 100);
    return `<tr>
      <td>${i + 1}</td>
      <td>${escapeHtml(p.stock)}</td>
      <td class="mono">${escapeHtml(p.ticker)}</td>
      <td>${prob.toFixed(4)}
        <div class="prob-bar"><div class="prob-fill" style="width:${pct}%;background:${up ? "var(--green)" : "var(--red)"}"></div></div>
      </td>
      <td class="${up ? "sig-up" : "sig-down"}">${up ? "▲ UP" : "▼ DOWN"}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="5" class="muted">No predictions.</td></tr>`;
}

// ---------------- Init ----------------
loadStatus();
