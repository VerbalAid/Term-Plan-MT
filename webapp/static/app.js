const form = document.getElementById("search-form");
const contextForm = document.getElementById("context-form");
const input = document.getElementById("term-input");
const contextInput = document.getElementById("context-input");
const targetInput = document.getElementById("target-input");
const submitBtn = document.getElementById("submit-btn");
const contextSubmitBtn = document.getElementById("context-submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const healthDot = document.getElementById("health-dot");
const healthText = document.getElementById("health-text");
const panelTerm = document.getElementById("panel-term");
const panelContext = document.getElementById("panel-context");
const modeTabs = document.querySelectorAll(".mode-tab");

let lang = "auto";
let mode = "term";

const placeholders = {
  auto: "French or English — e.g. pneumopathie inflammatoire, pembrolizumab…",
  fr: "e.g. pneumopathie inflammatoire, hypothyroïdie…",
  en: "e.g. immune-mediated pneumonitis, pembrolizumab…",
};

const matchLabels = {
  exact: "Exact",
  fuzzy: "Fuzzy",
  semantic: "Semantic",
  context_llm: "In context",
  none: "No match",
};

const pipeSteps = document.querySelectorAll(".pipe-step");

function activeLangButtons() {
  return mode === "context"
    ? document.querySelectorAll(".context-lang .lang-btn")
    : document.querySelectorAll("#panel-term .lang-btn");
}

function setLang(next) {
  lang = next;
  activeLangButtons().forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));
  if (mode === "term") input.placeholder = placeholders[lang] || placeholders.auto;
}

activeLangButtons().forEach((btn) => {
  btn.addEventListener("click", () => setLang(btn.dataset.lang));
});

document.querySelectorAll(".context-lang .lang-btn").forEach((btn) => {
  btn.addEventListener("click", () => setLang(btn.dataset.lang));
});

document.querySelectorAll("#panel-term .lang-btn").forEach((btn) => {
  btn.addEventListener("click", () => setLang(btn.dataset.lang));
});

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    modeTabs.forEach((t) => {
      const on = t === tab;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    const termOn = mode === "term";
    panelTerm.hidden = !termOn;
    panelTerm.classList.toggle("hidden", !termOn);
    panelContext.hidden = termOn;
    panelContext.classList.toggle("hidden", termOn);
    hideStatus();
    resultsEl.hidden = true;
    if (termOn) input.focus();
    else contextInput.focus();
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
    const data = await readJsonResponse(res);
    if (data.status === "ok") {
      healthDot.className = "health-dot ok";
      const sem = data.semantic_ready || {};
      const semNote = data.semantic_disabled
        ? " · semantic off (low RAM)"
        : sem.fr || sem.en
          ? " · semantic ready"
          : " · semantic loads on first use";
      const ctxNote = data.llm_configured
        ? " · context routing on"
        : " · context routing off";
      const cacheNote = data.cache_ready === false ? " · index loads on first search" : "";
      const labelsNote =
        data.labels_loaded != null
          ? `${Number(data.labels_loaded).toLocaleString()} labels`
          : "connected";
      healthText.textContent = `Neo4j · ${labelsNote}${cacheNote}${semNote}${ctxNote}`;
    } else {
      healthDot.className = "health-dot err";
      const hint = data.neo4j_target ? ` (${data.neo4j_target})` : "";
      healthText.textContent = (data.detail || "Neo4j unavailable") + hint;
    }
  } catch {
    healthDot.className = "health-dot err";
    healthText.textContent = "API unreachable";
  }
}

async function readJsonResponse(res) {
  const text = await res.text();
  if (!text.trim()) {
    throw new Error(
      res.status >= 500
        ? "Server error (empty response). The app may have timed out or restarted — try again in a minute."
        : "Empty response from server."
    );
  }
  try {
    return JSON.parse(text);
  } catch {
    const preview = text.slice(0, 80).replace(/\s+/g, " ");
    throw new Error(
      res.status >= 500 || res.status === 504
        ? "Request timed out or the server restarted (common on first semantic / In context use). Wait and retry."
        : `Server returned non-JSON (${res.status}): ${preview}`
    );
  }
}

function setLoading(on, which = "term") {
  const btn = which === "context" ? contextSubmitBtn : submitBtn;
  const label = btn.querySelector(".btn-label");
  const spin = btn.querySelector(".spinner");
  btn.disabled = on;
  spin.hidden = !on;
  label.hidden = on;
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
  modeTabs[0].click();
  input.value = term;
  if (searchLang && searchLang !== "auto") setLang(searchLang);
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

function renderConceptCard(data, { title = "Match", badgeExtra = "" } = {}) {
  if (data.match_type === "none" || !data.concept) {
    return `
      <article class="card">
        <div class="card-header">
          <div>
            <h2 class="concept-title">No match</h2>
            <p class="concept-fr">${esc(data.message || "Try another spelling or language.")}</p>
          </div>
          <span class="badge none">${esc(matchLabels.none)}</span>
        </div>
      </article>`;
  }

  const c = data.concept;
  const score = data.score != null ? `${data.score}%` : "—";
  const ql = data.query_lang || "fr";
  const enLine = c.en_label || c.name;
  const frLine = c.fr_label;
  const titlePrimary = ql === "fr" && frLine ? frLine : enLine;
  const subPrimary =
    ql === "fr"
      ? `<p class="concept-en">EN · ${esc(enLine)}</p>`
      : frLine
        ? `<p class="concept-fr">FR · ${esc(frLine)}</p>`
        : "";

  let alerts = "";
  if (data.message) {
    alerts += `<div class="alert info">${esc(data.message)}</div>`;
  }
  if (data.ambiguous) {
    alerts += `<div class="alert warn">${esc(data.message || "Ambiguous label.")}</div>`;
  }
  if (data.alternatives?.length) {
    alerts += `<div class="alert warn"><strong>Also:</strong> ${data.alternatives
      .map((a) => esc(a.name))
      .join(", ")}</div>`;
  }

  return `
    <article class="card">
      <div class="card-header">
        <div>
          <h2 class="concept-title">${esc(titlePrimary)}</h2>
          ${subPrimary}
        </div>
        <span class="badge ${esc(data.match_type)}">${esc(matchLabels[data.match_type] || data.match_type)} · ${score}${badgeExtra}</span>
      </div>
      <p class="card-subtitle">${esc(title)}</p>
      <div class="meta-row">
        <span>ID ${esc(c.id)}</span>
        <span>${esc(tierLabel(c.tier))}</span>
        <span>Level ${c.level ?? "—"}</span>
      </div>
      ${alerts}
      <h3 class="section-label">Hierarchy</h3>
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
    </article>`;
}

function renderResults(data) {
  resultsEl.hidden = false;
  setPipelineHighlight(data.match_type);
  resultsEl.innerHTML = renderConceptCard(data);
  wireNavigation(resultsEl, data.query_lang || "fr");
}

function renderCandidateList(candidates, queryLang) {
  if (!candidates?.length) {
    return '<p class="empty-col">No graph candidates retrieved.</p>';
  }
  return `<div class="candidate-list">${candidates
    .map((c) => {
      const label = queryLang === "fr" && c.fr_label ? c.fr_label : c.name;
      return `
      <div class="candidate-row">
        <div class="candidate-main">
          <strong>${esc(label)}</strong>
          <span class="candidate-meta">${esc(c.tier)} · ID ${esc(c.id)} · ${esc(c.match_source)} ${c.score != null ? `· ${c.score}%` : ""}</span>
        </div>
        <p class="candidate-sub">${esc(c.name)}${c.ancestor_summary ? ` · ${esc(c.ancestor_summary)}` : ""}</p>
      </div>`;
    })
    .join("")}</div>`;
}

function renderContextResults(payload) {
  resultsEl.hidden = false;
  pipeSteps.forEach((el) => el.classList.remove("active"));

  const ql = payload.query_lang || "fr";
  const llm = payload.llm || {};
  let html = "";

  if (llm.ok && !llm.abstain && llm.clinical_justification) {
    html += `
    <article class="card context-card">
      <div class="card-header">
        <div>
          <h2 class="concept-title">Resolution</h2>
          <p class="concept-fr">${esc(llm.confidence || "")} confidence</p>
        </div>
        <span class="badge context_llm">In context</span>
      </div>
      <div class="context-block">
        <h3 class="section-label">Clinical fit</h3>
        <p>${esc(llm.clinical_justification)}</p>
      </div>
      <div class="context-block">
        <h3 class="section-label">Register</h3>
        <p>${esc(llm.stylistic_analysis)}</p>
      </div>
    </article>`;
  } else if (llm.abstain) {
    html += `
    <article class="card context-card muted-card">
      <h2 class="concept-title">No clear match</h2>
      <p>${esc(llm.clinical_justification || "None of the candidates are clearly supported by the sentence.")}</p>
      ${llm.stylistic_analysis ? `<p class="concept-fr">${esc(llm.stylistic_analysis)}</p>` : ""}
    </article>`;
  } else if (!llm.configured) {
    html += `
    <article class="card context-card muted-card">
      <h2 class="concept-title">Candidates only</h2>
      <p>${esc(llm.message || "Context routing is not configured on the server.")}</p>
    </article>`;
  } else if (!llm.ok) {
    html += `
    <article class="card context-card muted-card">
      <h2 class="concept-title">Routing unavailable</h2>
      <p>${esc(llm.message || llm.error || "Could not complete contextual routing.")}</p>
    </article>`;
  }

  if (payload.selected) {
    html += renderConceptCard(payload.selected, { title: "Selected" });
  } else if (payload.baseline?.concept) {
    html += renderConceptCard(payload.baseline, { title: "Term lookup" });
  }

  html += `
    <article class="card">
      <h2 class="concept-title">Candidates · ${payload.candidates?.length ?? 0}</h2>
      ${renderCandidateList(payload.candidates, ql)}
    </article>`;

  resultsEl.innerHTML = html;
  const navData = payload.selected || payload.baseline;
  if (navData?.concept) wireNavigation(resultsEl, ql);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const term = input.value.trim();
  if (!term) return;

  hideStatus();
  resultsEl.hidden = true;
  pipeSteps.forEach((el) => el.classList.remove("active"));
  setLoading(true, "term");
  showStatus("Searching… first lookup may take 1–2 minutes while the index loads.", false);

  try {
    const res = await fetch("/api/lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term, lang }),
    });
    const data = await readJsonResponse(res);
    if (!res.ok) {
      setPipelineHighlight("none");
      showStatus(data.detail || "Lookup failed", true);
      return;
    }
    hideStatus();
    renderResults(data);
  } catch (err) {
    setPipelineHighlight("none");
    showStatus(err.message || "Network error", true);
  } finally {
    setLoading(false, "term");
  }
});

contextForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const context_sentence = contextInput.value.trim();
  const target_term = targetInput.value.trim();
  if (!context_sentence || !target_term) {
    showStatus("Enter a sentence and the term to ground.", true);
    return;
  }

  hideStatus();
  resultsEl.hidden = true;
  setLoading(true, "context");
  showStatus("Resolving… first request may take 1–2 minutes while the index loads.", false);

  try {
    const res = await fetch("/api/context-lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context_sentence, target_term, lang }),
    });
    const data = await readJsonResponse(res);
    if (!res.ok) {
      showStatus(data.detail || "Context lookup failed", true);
      return;
    }
    hideStatus();
    renderContextResults(data);
  } catch (err) {
    showStatus(err.message || "Network error", true);
  } finally {
    setLoading(false, "context");
  }
});

checkHealth();
setInterval(checkHealth, 60_000);
