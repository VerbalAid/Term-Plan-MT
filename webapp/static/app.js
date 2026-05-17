const form = document.getElementById("search-form");
const input = document.getElementById("term-input");
const submitBtn = document.getElementById("submit-btn");
const btnLabel = submitBtn.querySelector(".btn-label");
const spinner = submitBtn.querySelector(".spinner");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const healthDot = document.getElementById("health-dot");
const healthText = document.getElementById("health-text");
const langBtns = document.querySelectorAll(".lang-btn");

let lang = "fr";

const placeholders = {
  fr: "e.g. pneumopathie inflammatoire, hypothyroïdie…",
  en: "e.g. immune-mediated pneumonitis, pembrolizumab…",
};

langBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    lang = btn.dataset.lang;
    langBtns.forEach((b) => b.classList.toggle("active", b === btn));
    input.placeholder = placeholders[lang];
    input.focus();
  });
});

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      healthDot.className = "health-dot ok";
      healthText.textContent = `Neo4j connected · ${data.labels_loaded?.toLocaleString() ?? "?"} labels`;
    } else {
      healthDot.className = "health-dot err";
      healthText.textContent = data.detail || "Neo4j unavailable";
    }
  } catch {
    healthDot.className = "health-dot err";
    healthText.textContent = "API unreachable";
  }
}

function setLoading(on) {
  submitBtn.disabled = on;
  spinner.hidden = !on;
  btnLabel.hidden = on;
}

function showStatus(msg, isError = false) {
  statusEl.hidden = false;
  statusEl.className = `status${isError ? " error" : ""}`;
  statusEl.textContent = msg;
}

function hideStatus() {
  statusEl.hidden = true;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}

function tierLabel(tier) {
  const map = {
    SOC: "System Organ Class",
    HLGT: "HLGT",
    HLT: "HLT",
    PT: "Preferred Term",
    LLT: "Lowest Level Term",
  };
  return map[tier] || tier;
}

function renderPills(items) {
  if (!items?.length) {
    return '<p class="empty-col">None</p>';
  }
  return `<div class="pill-list">${items
    .map(
      (c) => `
    <div class="pill">
      ${esc(c.name)}
      <small>${esc(c.tier)}${c.fr_label ? ` · ${esc(c.fr_label)}` : ""}</small>
    </div>`
    )
    .join("")}</div>`;
}

function renderHierarchy(ancestors) {
  if (!ancestors?.length) return "";
  return `<div class="hierarchy">${ancestors
    .map((n, i) => {
      const isLast = i === ancestors.length - 1;
      const cls = isLast ? "hier-node current" : "hier-node";
      const sep = i < ancestors.length - 1 ? '<span class="hier-sep">›</span>' : "";
      return `<span class="${cls}">${esc(n.tier)}: ${esc(n.name)}</span>${sep}`;
    })
    .join("")}</div>`;
}

function renderResults(data) {
  resultsEl.hidden = false;

  if (data.match_type === "none" || !data.concept) {
    resultsEl.innerHTML = `
      <article class="card">
        <div class="card-header">
          <div>
            <h2 class="concept-title">No match</h2>
            <p class="concept-fr">${esc(data.message || "Try another spelling or language.")}</p>
          </div>
          <span class="badge none">No match</span>
        </div>
      </article>`;
    return;
  }

  const c = data.concept;
  const score = data.score != null ? `${data.score}%` : "—";

  let alerts = "";
  if (data.ambiguous) {
    alerts += `<div class="alert warn">${esc(data.message || "Multiple MedDRA concepts share this French label.")}</div>`;
  }
  if (data.alternatives?.length) {
    alerts += `<div class="alert warn"><strong>Other matches:</strong> ${data.alternatives
      .map((a) => esc(a.name))
      .join(", ")}</div>`;
  }

  resultsEl.innerHTML = `
    <article class="card">
      <div class="card-header">
        <div>
          <h2 class="concept-title">${esc(c.name)}</h2>
          ${c.fr_label ? `<p class="concept-fr">${esc(c.fr_label)}</p>` : ""}
        </div>
        <span class="badge ${esc(data.match_type)}">${esc(data.match_type)} · ${score}</span>
      </div>
      <div class="meta-row">
        <span>ID ${esc(c.id)}</span>
        <span>${esc(tierLabel(c.tier))}</span>
        <span>Level ${c.level ?? "—"}</span>
      </div>
      ${alerts}
      ${renderHierarchy(data.ancestors)}
      <div class="cols">
        <div class="col">
          <h3>Parents (broader)</h3>
          ${renderPills(data.parents)}
        </div>
        <div class="col">
          <h3>Children (narrower)</h3>
          ${renderPills(data.children)}
        </div>
      </div>
    </article>`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const term = input.value.trim();
  if (!term) return;

  hideStatus();
  resultsEl.hidden = true;
  setLoading(true);

  try {
    const res = await fetch("/api/lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term, lang }),
    });
    const data = await res.json();
    if (!res.ok) {
      showStatus(data.detail || "Lookup failed", true);
      return;
    }
    renderResults(data);
  } catch (err) {
    showStatus(err.message || "Network error", true);
  } finally {
    setLoading(false);
  }
});

checkHealth();
