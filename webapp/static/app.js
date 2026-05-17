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
const pipeSteps = document.querySelectorAll(".pipe-step");

let lang = "auto";

const placeholders = {
  auto: "French or English — e.g. pneumopathie inflammatoire, pembrolizumab…",
  fr: "e.g. pneumopathie inflammatoire, hypothyroïdie…",
  en: "e.g. immune-mediated pneumonitis, pembrolizumab…",
};

const matchLabels = {
  exact: "Exact Match",
  fuzzy: "Fuzzy Match (RapidFuzz)",
  semantic: "Semantic Match (Vector)",
  none: "No Match",
};

langBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    lang = btn.dataset.lang;
    langBtns.forEach((b) => b.classList.toggle("active", b === btn));
    input.placeholder = placeholders[lang] || placeholders.auto;
    input.focus();
  });
});

function setPipelineHighlight(matchType) {
  pipeSteps.forEach((el) => {
    el.classList.toggle("active", el.dataset.step === matchType);
  });
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      healthDot.className = "health-dot ok";
      const sem = data.semantic_ready || {};
      const semNote =
        sem.fr || sem.en ? " · semantic ready" : " · semantic loads on first use";
      healthText.textContent = `Neo4j · ${data.labels_loaded?.toLocaleString() ?? "?"} FR labels${semNote}`;
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
  const box = document.createElement("div");
  box.textContent = s ?? "";
  return box.innerHTML;
}

function tierLabel(tier) {
  const map = {
    SOC: "System Organ Class",
    HLGT: "High Level Group Term",
    HLT: "High Level Term",
    PT: "Preferred Term",
    LLT: "Lowest Level Term",
  };
  return map[tier] || tier;
}

function pillLang(c, preferredLang) {
  if (preferredLang === "fr" && c.fr_label) return { term: c.fr_label, lang: "fr" };
  return { term: c.name || c.en_label, lang: "en" };
}

function lookupTerm(term, searchLang) {
  input.value = term;
  if (searchLang && searchLang !== "auto") {
    lang = searchLang;
    langBtns.forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));
    input.placeholder = placeholders[lang];
  }
  form.requestSubmit();
}

function renderPills(items, preferredLang) {
  if (!items?.length) {
    return '<p class="empty-col">None at this level</p>';
  }
  return `<div class="pill-list">${items
    .map((c) => {
      const { term, lang: pl } = pillLang(c, preferredLang);
      const sub = preferredLang === "fr" ? c.name : c.fr_label || "";
      return `
    <button type="button" class="pill" data-term="${esc(term)}" data-lang="${pl}">
      ${esc(preferredLang === "fr" ? c.fr_label || c.name : c.name)}
      <small>${esc(c.tier)}${sub ? ` · ${esc(sub)}` : ""}</small>
    </button>`;
    })
    .join("")}</div>`;
}


function renderHierarchy(ancestors, queryLang) {
  if (!ancestors?.length) return "";
  return `<div class="hierarchy">${ancestors
    .map((n, i) => {
      const isLast = i === ancestors.length - 1;
      const cls = isLast ? "hier-node current" : "hier-node hier-btn";
      const { term, lang: hl } = pillLang(n, queryLang);
      const sep = i < ancestors.length - 1 ? '<span class="hier-sep">›</span>' : "";
      const attrs = isLast
        ? ""
        : ` data-term="${esc(term)}" data-lang="${hl}" role="button" tabindex="0"`;
      return `<span class="${cls}"${attrs}>${esc(n.tier)}: ${esc(n.name)}</span>${sep}`;
    })
    .join("")}</div>`;
}

function wireNavigation(root, queryLang) {
  root.querySelectorAll(".pill[data-term], .hier-btn[data-term]").forEach((btn) => {
    btn.addEventListener("click", () => {
      lookupTerm(btn.dataset.term, btn.dataset.lang || queryLang);
    });
    btn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        lookupTerm(btn.dataset.term, btn.dataset.lang || queryLang);
      }
    });
  });
}

function renderResults(data) {
  resultsEl.hidden = false;
  setPipelineHighlight(data.match_type);

  if (data.match_type === "none" || !data.concept) {
    resultsEl.innerHTML = `
      <article class="card">
        <div class="card-header">
          <div>
            <h2 class="concept-title">No match</h2>
            <p class="concept-fr">${esc(data.message || "Try another spelling or language.")}</p>
          </div>
          <span class="badge none">${esc(matchLabels.none)}</span>
        </div>
      </article>`;
    return;
  }

  const c = data.concept;
  const score = data.score != null ? `${data.score}%` : "—";
  const ql = data.query_lang || "fr";
  const enLine = c.en_label || c.name;
  const frLine = c.fr_label;

  let alerts = "";
  if (data.ambiguous) {
    alerts += `<div class="alert warn">${esc(data.message || "Ambiguous label.")}</div>`;
  }
  if (data.alternatives?.length) {
    alerts += `<div class="alert warn"><strong>Also:</strong> ${data.alternatives
      .map((a) => esc(a.name))
      .join(", ")}</div>`;
  }

  const titlePrimary = ql === "fr" && frLine ? frLine : enLine;
  const subPrimary =
    ql === "fr"
      ? `<p class="concept-en">EN · ${esc(enLine)}</p>`
      : frLine
        ? `<p class="concept-fr">FR · ${esc(frLine)}</p>`
        : "";

  resultsEl.innerHTML = `
    <article class="card">
      <div class="card-header">
        <div>
          <h2 class="concept-title">${esc(titlePrimary)}</h2>
          ${subPrimary}
        </div>
        <span class="badge ${esc(data.match_type)}">${esc(matchLabels[data.match_type] || data.match_type)} · ${score}</span>
      </div>
      <div class="meta-row">
        <span>ID ${esc(c.id)}</span>
        <span>${esc(tierLabel(c.tier))}</span>
        <span>Level ${c.level ?? "—"}</span>
        <span>Query · ${esc(ql.toUpperCase())}</span>
      </div>
      ${alerts}
      <h3 class="section-label">Hierarchy (SOC → match)</h3>
      ${renderHierarchy(data.ancestors, ql)}
      <div class="cols">
        <div class="col">
          <h3>Parents (broader)</h3>
          ${renderPills(data.parents, "en")}
        </div>
        <div class="col">
          <h3>Children (narrower)</h3>
          ${renderPills(data.children, "en")}
        </div>
      </div>
      <p class="hint pill-hint">Click any parent, child, or ancestor to explore the graph</p>
    </article>`;

  wireNavigation(resultsEl, ql);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const term = input.value.trim();
  if (!term) return;

  hideStatus();
  resultsEl.hidden = true;
  pipeSteps.forEach((el) => el.classList.remove("active"));
  setLoading(true);

  try {
    const res = await fetch("/api/lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term, lang }),
    });
    const data = await res.json();
    if (!res.ok) {
      setPipelineHighlight("none");
      showStatus(data.detail || "Lookup failed", true);
      return;
    }
    renderResults(data);
  } catch (err) {
    setPipelineHighlight("none");
    showStatus(err.message || "Network error", true);
  } finally {
    setLoading(false);
  }
});

checkHealth();
setInterval(checkHealth, 60_000);
