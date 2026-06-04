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
  if (t.dataset.tab === "predictions") { loadPredictions(); loadPredDates(); }
  if (t.dataset.tab === "advisor") loadAdvisorStatus();
}));

// ---------------- Dashboard ----------------
async function loadStatus() {
  try {
    const s = await api.get("/api/status");
    setBusy(s.busy);
    const card = (label, value, sub, cls = "") =>
      `<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub || ""}</div></div>`;
    $("#status-cards").innerHTML =
      card("Model Name", s.project, "active configuration", "modelname") +
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
  if (pill) {
    pill.textContent = busy ? "● running" : "● idle";
    pill.className = "pill " + (busy ? "pill-busy" : "pill-idle");
  }
  const dash = $("#dash-busy");
  if (dash) {
    dash.textContent = busy ? "● running" : "● idle";
    dash.className = "pill " + (busy ? "pill-busy" : "pill-idle");
  }
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
$$("[data-stage]").forEach((b) => b.addEventListener("click", () => {
  const stage = b.dataset.stage;
  const opts = {};
  // Only the predict stages take a target date.
  if (stage === "predict" || stage === "predict-pipeline") {
    opts.asOfDate = $("#dash-pred-date")?.value || $("#pred-date")?.value || undefined;
  }
  runStage(stage, b.dataset.mode, opts);
}));

let lastJobs = [];

async function runStage(stage, mode, opts = {}) {
  try {
    const body = { stage, mode: mode || null };
    if (opts.asOfDate) body.as_of_date = opts.asOfDate;
    const job = await api.post("/api/pipeline/run", body);
    selectedJob = job.id;
    await loadJobs();
    startPolling();
  } catch (e) { alert(e.message); }
}

async function loadJobs() {
  const jobs = await api.get("/api/pipeline/jobs");
  lastJobs = jobs;
  setBusy(jobs.some((j) => j.status === "running"));

  const tbody = $("#jobs-tbody");
  if (tbody) {
    tbody.innerHTML = jobs.map((j) => `
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
  }

  renderDashJobs();
  if (selectedJob) loadLog();
}

function fmtDuration(d) {
  if (d == null) return "—";
  const s = Math.round(d);
  if (s < 60) return s + "s";
  return Math.floor(s / 60) + "m " + (s % 60) + "s";
}

function renderDashJobs() {
  const tbody = $("#dash-jobs-tbody");
  if (!tbody) return;
  const jobs = lastJobs.slice(0, 6);
  tbody.innerHTML = jobs.map((j) => `
    <tr class="clickable ${j.id === selectedJob ? "sel" : ""}" data-id="${j.id}">
      <td>${j.stage}${j.mode ? " <span class='muted'>(" + j.mode + ")</span>" : ""}</td>
      <td><span class="status-badge s-${j.status}">${j.status}</span></td>
      <td>${fmtTime(j.started)}</td>
      <td>${fmtDuration(j.duration)}</td>
    </tr>`).join("") || `<tr><td colspan="4" class="muted">No jobs yet.</td></tr>`;

  $$("#dash-jobs-tbody tr.clickable").forEach((tr) => tr.addEventListener("click", () => {
    selectedJob = tr.dataset.id;
    if ($("#jobs-tbody")) loadJobs();
    renderDashJobs();
    loadLog();
  }));
}

async function loadLog() {
  if (!selectedJob) return;
  // Don't request a job that's no longer in the list (e.g. server restarted) —
  // it would 404 on every poll. Drop the stale selection instead.
  if (lastJobs.length && !lastJobs.some((j) => j.id === selectedJob)) {
    selectedJob = null;
    return;
  }
  try {
    const res = await api.get(`/api/pipeline/jobs/${selectedJob}/log`);
    const idEl = $("#log-job-id");
    if (idEl) idEl.textContent = "· " + selectedJob;
    const box = $("#job-log");
    if (box) {
      box.textContent = res.log || "(no output yet)";
      box.scrollTop = box.scrollHeight;
    }
    const dashId = $("#dash-log-id");
    if (dashId) dashId.textContent = "· " + selectedJob;
    const dashBox = $("#dash-log");
    if (dashBox) {
      dashBox.textContent = res.log || "(no output yet)";
      dashBox.scrollTop = dashBox.scrollHeight;
    }
  } catch (e) {
    // Job vanished server-side (e.g. restart) — clear it so we stop polling/404ing.
    selectedJob = null;
  }
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
$("#run-pred").addEventListener("click", () =>
  runStage("predict-pipeline", null, { asOfDate: $("#pred-date")?.value || undefined })
);

// Populate the prediction target-date pickers and keep them in sync.
function _setDateInput(el, info) {
  if (!el) return;
  if (info && info.available) {
    el.min = info.min_date || "";
    el.max = info.max_date || "";
    if (info.default_date) el.value = info.default_date;
    el.disabled = false;
  } else {
    el.value = "";
    el.disabled = true;
  }
}

async function loadPredDates() {
  let info = null;
  try {
    info = await api.get("/api/predictions/dates");
  } catch (_) {
    info = null;
  }
  const predDate = $("#pred-date");
  const dashDate = $("#dash-pred-date");
  _setDateInput(predDate, info);
  _setDateInput(dashDate, info);
  if (predDate && dashDate && !predDate._synced) {
    predDate._synced = true;
    predDate.addEventListener("change", () => { dashDate.value = predDate.value; });
    dashDate.addEventListener("change", () => { predDate.value = dashDate.value; });
  }
}

// Prettify a raw CSV column name into a header label.
function prettyCol(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Columns shown in the prediction tables. "Date" duplicates as_of_date, so hide it.
const HIDDEN_PRED_COLS = new Set(["Date"]);
function predCols(rows) {
  return rows.length ? Object.keys(rows[0]).filter((c) => !HIDDEN_PRED_COLS.has(c)) : [];
}

// Render a single cell, with special formatting for known columns.
function renderCell(key, value, up) {
  if (key === "up_probability") {
    const prob = parseFloat(value);
    if (!isNaN(prob)) {
      const pct = Math.round(prob * 100);
      return `<td>${prob.toFixed(4)}
        <div class="prob-bar"><div class="prob-fill" style="width:${pct}%;background:${up ? "var(--green)" : "var(--red)"}"></div></div>
      </td>`;
    }
  }
  if (key === "signal") {
    return `<td class="${up ? "sig-up" : "sig-down"}">${up ? "▲ UP" : "▼ DOWN"}</td>`;
  }
  if (key === "ticker") {
    return `<td class="mono">${escapeHtml(value)}</td>`;
  }
  return `<td>${escapeHtml(value)}</td>`;
}

async function loadPredictions() {
  const data = await api.get("/api/predictions");
  const rows = data.predictions || [];
  const meta = $("#pred-meta");
  if (meta) {
    if (data.count) {
      const first = rows[0];
      const target = first.target_date || first.as_of_date;
      meta.textContent = `Signals for ${target} · data through ${first.as_of_date} · ${data.count} stocks`;
    } else {
      meta.textContent = "No predictions yet — run the predict pipeline.";
    }
  }

  // Derive columns from the CSV headers (Date is hidden — see predCols).
  // A live "Current Situation" column is appended (filled async, see below).
  const cols = predCols(rows);
  $("#pred-thead").innerHTML = rows.length
    ? `<tr><th>#</th>${cols.map((c) => `<th>${escapeHtml(prettyCol(c))}</th>`).join("")}<th>Current Situation</th></tr>`
    : "";

  $("#pred-tbody").innerHTML = rows.map((p, i) => {
    const up = (p.signal || "").toUpperCase() === "UP";
    const cells = cols.map((c) => renderCell(c, p[c], up)).join("");
    const live = `<td class="live-cell" data-ticker="${escapeHtml(p.ticker || "")}"><span class="muted">…</span></td>`;
    return `<tr><td>${i + 1}</td>${cells}${live}</tr>`;
  }).join("") || `<tr><td class="muted">No predictions.</td></tr>`;

  if (rows.length) loadLiveSituation();
}

// Fetch live prices and fill the "Current Situation" column (vs target-day open).
async function loadLiveSituation() {
  let data;
  try {
    data = await api.get("/api/predictions/live");
  } catch (_) {
    return;
  }
  if (!data || !data.available) {
    $$("#pred-tbody .live-cell").forEach((td) => { td.innerHTML = `<span class="muted">—</span>`; });
    return;
  }
  const rows = data.rows || {};
  $$("#pred-tbody .live-cell").forEach((td) => {
    td.innerHTML = renderLiveSituation(rows[td.dataset.ticker]);
  });
}

function renderLiveSituation(info) {
  if (!info || info.status === "pending") return `<span class="muted" title="market not open yet for the target day">— pending</span>`;
  if (info.status === "error" || info.change_pct == null) return `<span class="muted">n/a</span>`;
  const up = info.change_pct >= 0;
  const arrow = up ? "▲" : "▼";
  const sign = info.change_pct > 0 ? "+" : "";
  const match = info.matched === true
    ? ` <span class="live-ok" title="matches the signal">✓</span>`
    : info.matched === false
      ? ` <span class="live-bad" title="against the signal">✗</span>`
      : "";
  const dot = info.status === "live" ? `<span class="live-dot" title="live (today)"></span>` : "";
  return `<span class="${up ? "sig-up" : "sig-down"}">${arrow} ${sign}${info.change_pct}%</span>${match}` +
    `<div class="live-sub muted">${dot}open ${fmtMoney(info.open)} → ${fmtMoney(info.current)}</div>`;
}

// Dashboard: compact top-N predictions table.
async function loadDashPredictions() {
  const thead = $("#dash-pred-thead");
  const tbody = $("#dash-pred-tbody");
  if (!thead && !tbody) return;
  try {
    const data = await api.get("/api/predictions");
    const rows = (data.predictions || []).slice(0, 8);
    const cols = predCols(rows);
    if (thead) {
      thead.innerHTML = rows.length
        ? `<tr><th>#</th>${cols.map((c) => `<th>${escapeHtml(prettyCol(c))}</th>`).join("")}</tr>`
        : "";
    }
    if (tbody) {
      tbody.innerHTML = rows.map((p, i) => {
        const up = (p.signal || "").toUpperCase() === "UP";
        const cells = cols.map((c) => renderCell(c, p[c], up)).join("");
        return `<tr><td>${i + 1}</td>${cells}</tr>`;
      }).join("") || `<tr><td class="muted">No predictions yet — run the predict pipeline.</td></tr>`;
    }
  } catch (e) {
    if (tbody) tbody.innerHTML = `<tr><td class="muted">Could not load predictions: ${escapeHtml(e.message)}</td></tr>`;
  }
}

$("#dash-pred-refresh")?.addEventListener("click", loadDashPredictions);

// ---------------- AI Advisor ----------------
const CUR_SYMBOL = { INR: "₹", USD: "$", EUR: "€", GBP: "£" };
let advCurrency = "INR";

function fmtMoney(n) {
  const code = advCurrency || "INR";
  try {
    return new Intl.NumberFormat(code === "INR" ? "en-IN" : "en-US",
      { style: "currency", currency: code, maximumFractionDigits: 0 }).format(n);
  } catch {
    return `${CUR_SYMBOL[code] || ""}${Number(n).toLocaleString()}`;
  }
}

// Model confidence (up_probability) as a percentage; "—" for CASH/unknown.
function fmtConfidence(p) {
  if (p == null || isNaN(p)) return `<span class="muted">—</span>`;
  return `${Math.round(p * 100)}%`;
}

// Current price + intraday move from open; "—" when no live data (e.g. CASH).
function fmtCurrentPrice(price, changePct) {
  if (price == null || isNaN(price)) return `<span class="muted">—</span>`;
  let move = "";
  if (changePct != null && !isNaN(changePct)) {
    const up = changePct >= 0;
    move = `<div class="live-sub ${up ? "sig-up" : "sig-down"}">${up ? "▲" : "▼"} ${changePct > 0 ? "+" : ""}${changePct}%</div>`;
  }
  return `${fmtMoney(price)}${move}`;
}

// Intraday exit level (target/stop-loss) as a price + the % offset from entry.
function fmtExit(price, pct, dir) {
  if (price == null || isNaN(price)) return `<span class="muted">—</span>`;
  const cls = dir === "up" ? "sig-up" : "sig-down";
  const sign = dir === "up" ? "+" : "−";
  const pctTxt = (pct != null && !isNaN(pct)) ? `<div class="live-sub ${cls}">${sign}${pct}%</div>` : "";
  return `<span class="${cls}">${fmtMoney(price)}</span>${pctTxt}`;
}

async function loadAdvisorStatus() {
  const statusEl = $("#adv-status");
  const runBtn = $("#adv-run");
  try {
    const s = await api.get("/api/advisor/status");
    advCurrency = s.currency || "INR";
    $("#adv-currency").textContent = CUR_SYMBOL[advCurrency] || advCurrency;
    const bits = [];
    bits.push(s.deps_installed ? "deps ✓" : "deps ✗ (pip install)");
    bits.push(`${s.llm.provider}:${s.llm.model || "?"} ${s.llm.key_present ? "✓" : "✗ key"}`);
    bits.push(`news ${s.tavily.key_present ? "✓" : "off (no Tavily key)"}`);
    statusEl.textContent = bits.join(" · ");
    statusEl.className = s.ready ? "muted ok" : "muted bad";
    if (runBtn) {
      runBtn.disabled = !s.ready;
      runBtn.title = s.ready ? "" : "Install deps and set the LLM API key first.";
    }
    const newsBox = $("#adv-news");
    if (newsBox && !s.tavily.key_present) newsBox.checked = false;
  } catch (e) {
    statusEl.textContent = e.message;
    statusEl.className = "muted bad";
  }
}

async function runAdvisor() {
  const budget = parseFloat($("#adv-budget").value);
  const errEl = $("#adv-error");
  errEl.textContent = "";
  if (!budget || budget <= 0) { errEl.textContent = "Enter a positive amount to invest."; return; }

  $("#adv-loading").hidden = false;
  $("#adv-plans").innerHTML = "";
  $("#adv-news-wrap").innerHTML = "";
  $("#adv-outlook").innerHTML = "";
  $("#adv-meta").textContent = "";
  const runBtn = $("#adv-run");
  runBtn.disabled = true;
  try {
    const res = await api.post("/api/advisor/plan", {
      budget,
      include_news: $("#adv-news").checked,
    });
    advCurrency = res.currency || advCurrency;
    renderAdvisor(res);
  } catch (e) {
    errEl.textContent = e.message;
  } finally {
    $("#adv-loading").hidden = true;
    runBtn.disabled = false;
  }
}

function renderAdvisor(res) {
  const gen = res.generated_with || {};
  $("#adv-meta").textContent =
    `Plans for ${res.target_date || "next session"} · data through ${res.as_of_date || "—"} · ` +
    `${gen.provider || ""} ${gen.model || ""} · news ${res.news_enabled ? "on" : "off"}`;

  $("#adv-outlook").innerHTML = res.market_outlook
    ? `<h3>Market outlook</h3><p>${escapeHtml(res.market_outlook)}</p>` : "";

  // Plan cards — one per row, each collapsible (open the first by default).
  $("#adv-plans").innerHTML = (res.plans || []).map((p, i) => {
    const rows = (p.allocations || []).map((a) => `
      <tr>
        <td>${escapeHtml(a.stock)}</td>
        <td class="mono">${escapeHtml(a.ticker)}</td>
        <td class="num">${fmtConfidence(a.up_probability)}</td>
        <td class="num">${fmtCurrentPrice(a.entry_price, a.change_pct)}</td>
        <td class="num">${fmtExit(a.target_price, a.target_pct, "up")}</td>
        <td class="num">${fmtExit(a.stoploss_price, a.stoploss_pct, "down")}</td>
        <td class="num">${fmtMoney(a.amount)}<div class="live-sub muted">${a.percent}%</div></td>
        <td class="adv-reason">${escapeHtml(a.reason || "")}</td>
      </tr>`).join("") || `<tr><td colspan="8" class="muted">No allocations.</td></tr>`;
    const thr = (p.threshold != null) ? Math.round(p.threshold * 100) : null;
    return `
      <details class="plan-card plan-${p.key}"${i === 0 ? " open" : ""}>
        <summary class="plan-acc-head">
          <span class="plan-caret">▸</span>
          <span class="plan-title-text">${escapeHtml(p.title)}</span>
          <span class="risk-badge risk-${p.risk_level.toLowerCase()}">${escapeHtml(p.risk_level)} risk</span>
          <span class="plan-total-inline">${fmtMoney(p.total)}</span>
        </summary>
        <div class="plan-body">
          <p class="plan-obj muted">${escapeHtml(p.objective)}${thr != null ? ` · <span class="plan-thr">picks with model confidence &gt; ${thr}%</span>` : ""}</p>
          ${p.summary ? `<p class="plan-summary">${escapeHtml(p.summary)}</p>` : ""}
          ${p.expected_return ? `<p class="plan-exp"><strong>Expected:</strong> ${escapeHtml(p.expected_return)}</p>` : ""}
          <table class="data-table plan-table">
            <thead><tr><th>Stock</th><th>Ticker</th><th class="num">Confidence</th><th class="num">Entry (buy)</th><th class="num">Target (sell)</th><th class="num">Stop-loss</th><th class="num">Amount</th><th>Why</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
          <p class="plan-trade-note muted">Intraday: buy near <em>Entry</em>, book profit at <em>Target</em> (+${p.target_pct}%), exit/cut at <em>Stop-loss</em> (−${p.stoploss_pct}%).</p>
          <div class="plan-total">Total invested: <strong>${fmtMoney(p.total)}</strong></div>
        </div>
      </details>`;
  }).join("");

  // News section.
  let news = "";
  if ((res.market_news || []).length) {
    news += `<h3>Market news</h3><ul class="news-list">` +
      res.market_news.map(newsItem).join("") + `</ul>`;
  }
  const stockNews = res.stock_news || {};
  const tickers = Object.keys(stockNews);
  if (tickers.length) {
    news += `<h3>Per-stock news</h3>` + tickers.map((tk) =>
      `<details class="news-stock"><summary>${escapeHtml(tk)} <span class="muted">(${stockNews[tk].length})</span></summary>
        <ul class="news-list">${stockNews[tk].map(newsItem).join("")}</ul></details>`).join("");
  }
  $("#adv-news-wrap").innerHTML = news;
}

function newsItem(n) {
  const title = escapeHtml(n.title || "(untitled)");
  const link = n.url ? `<a href="${escapeHtml(n.url)}" target="_blank" rel="noopener">${title}</a>` : title;
  return `<li>${link}<div class="news-snippet muted">${escapeHtml(n.content || "")}</div></li>`;
}

$("#adv-run")?.addEventListener("click", runAdvisor);

// ---------------- Init ----------------
loadStatus();
loadJobs();
loadDashPredictions();
loadPredDates();
setInterval(loadStatus, 5000);
setInterval(loadJobs, 5000);
setInterval(loadDashPredictions, 15000);
