// ============================================================
// Predict-Your-Own-Model — Pyodide-powered in-browser predictor
// ============================================================

const STATE = {
  data: null,
  pyodide: null,
  benchIdx: {},   // bench_id -> column index in data.json
  ready: false,
};

const BEST_PREDICTIVE_PROBES = [
  'gpqa_diamond', 'hle', 'aime_2024', 'mmlu_pro', 'arc_agi_1',
  'arc_agi_2', 'aider_polyglot_diff', 'livecodebench', 'terminal_bench',
  'swe_bench_verified'
];

const LOW_COST_PROBES = [
  'gpqa_diamond', 'mmlu_pro', 'aider_polyglot_diff', 'matharena_apex_2025',
  'hmmt_nov_2025', 'bullshit_pushback', 'math_500', 'aime_2025',
  'arena_hard', 'ifbench'
];

// -----------------------------------------------------------
// Loader progress
// -----------------------------------------------------------
function setProgress(pct, sub) {
  const bar = document.getElementById('progress-bar');
  if (bar) bar.style.width = pct + '%';
  if (sub) document.getElementById('loader-sub').textContent = sub;
}

// -----------------------------------------------------------
// Boot: load data.json + Pyodide + predictor.py in parallel
// -----------------------------------------------------------
async function boot() {
  try {
    setProgress(5, 'Fetching benchmark matrix…');
    const dataPromise = fetch('../data.json?v=7').then(r => r.json());

    setProgress(15, 'Loading Python runtime…');
    const pyodide = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/'
    });
    STATE.pyodide = pyodide;
    setProgress(60, 'Loading numerical backend…');
    await pyodide.loadPackage(['numpy']);

    setProgress(80, 'Loading predictor…');
    const predictorSrc = await fetch('predictor.py?v=3').then(r => r.text());
    pyodide.FS.writeFile('predictor.py', predictorSrc);
    pyodide.runPython('import predictor');

    setProgress(95, 'Indexing benchmarks…');
    STATE.data = await dataPromise;
    STATE.benchIdx = {};
    STATE.data.benchmarks.forEach((b, i) => { STATE.benchIdx[b.id] = i; });

    // Pre-upload the observed matrix to Python (avoid reserializing on every predict)
    pyodide.globals.set('M_OBS_JSON', JSON.stringify(STATE.data.observed));
    pyodide.runPython(`
import json
M_OBS = json.loads(M_OBS_JSON)
del M_OBS_JSON
`);

    setProgress(100, 'Ready.');
    STATE.ready = true;
    setTimeout(() => {
      document.getElementById('loader').hidden = true;
      document.getElementById('form').hidden = false;
      buildForm();
    }, 350);
  } catch (e) {
    console.error(e);
    setProgress(0, 'Failed to load: ' + e.message);
  }
}

// -----------------------------------------------------------
// Form: dynamic score rows
// -----------------------------------------------------------
function buildForm() {
  setScoreRows(BEST_PREDICTIVE_PROBES.slice(0, 5));
  populateTargetBenchmark();
  renderStarterProbeSummary(BEST_PREDICTIVE_PROBES, 'Best predictive set');

  document.getElementById('use-best-probes').onclick = (e) => {
    e.preventDefault();
    setScoreRows(BEST_PREDICTIVE_PROBES.slice(0, 5));
    renderStarterProbeSummary(BEST_PREDICTIVE_PROBES, 'Best predictive set');
  };

  document.getElementById('use-low-cost-probes').onclick = (e) => {
    e.preventDefault();
    setScoreRows(LOW_COST_PROBES.slice(0, 5));
    renderStarterProbeSummary(LOW_COST_PROBES, 'Low-cost set');
  };

  document.getElementById('add-row').onclick = (e) => {
    e.preventDefault();
    addScoreRow();
  };
  document.getElementById('predict-btn').onclick = (e) => {
    e.preventDefault();
    runPrediction(false);
  };
  document.getElementById('predict-all-btn').onclick = (e) => {
    e.preventDefault();
    runPrediction(true);
  };
}

function setScoreRows(benchIds) {
  const rows = document.getElementById('score-rows');
  rows.innerHTML = '';
  benchIds.forEach(bid => addScoreRow(bid));
}

function renderStarterProbeSummary(probeIds, label) {
  const el = document.getElementById('starter-probes');
  const names = probeIds.slice(0, 5).map(benchNameById).join(', ');
  el.innerHTML = `<strong>${escapeHtml(label)}:</strong> ${escapeHtml(names)}.`;
}

function benchOptionsHTML(selectedId) {
  const opts = ['<option value="">-- Select benchmark --</option>'];
  for (const b of STATE.data.benchmarks) {
    const sel = b.id === selectedId ? ' selected' : '';
    opts.push(`<option value="${b.id}"${sel}>${b.name}</option>`);
  }
  return opts.join('');
}

function populateTargetBenchmark() {
  const target = document.getElementById('target-benchmark');
  const opts = ['<option value="">-- Choose one benchmark --</option>'];
  for (const b of STATE.data.benchmarks) {
    opts.push(`<option value="${b.id}">${b.name}</option>`);
  }
  target.innerHTML = opts.join('');
}

function addScoreRow(presetBenchId) {
  const rows = document.getElementById('score-rows');
  const div = document.createElement('div');
  div.className = 'score-row';
  div.innerHTML = `
    <select class="bench-sel">${benchOptionsHTML(presetBenchId)}</select>
    <input type="number" step="any" class="score-inp" placeholder="Score">
    <input type="text" class="source-inp" placeholder="Source URL or note">
    <button class="remove-btn" title="Remove">×</button>
  `;
  div.querySelector('.remove-btn').onclick = (e) => {
    e.preventDefault();
    div.remove();
  };
  rows.appendChild(div);
}

// -----------------------------------------------------------
// Prediction
// -----------------------------------------------------------
async function runPrediction(showAll) {
  const errEl = document.getElementById('error');
  errEl.hidden = true;

  const modelName = document.getElementById('model-name').value.trim() || 'your model';
  const targetBenchId = document.getElementById('target-benchmark').value;
  if (!showAll && !targetBenchId) {
    errEl.textContent = 'Please choose a target benchmark, or use All benchmarks.';
    errEl.hidden = false;
    return;
  }
  const rows = document.querySelectorAll('.score-row');
  const scores = {};
  let knownIds = new Set();
  const knownEntries = [];
  for (const r of rows) {
    const bid = r.querySelector('.bench-sel').value;
    const val = r.querySelector('.score-inp').value;
    const source = r.querySelector('.source-inp')?.value.trim() || '';
    if (!bid || val === '') continue;
    const v = parseFloat(val);
    if (!isFinite(v)) continue;
    const colIdx = STATE.benchIdx[bid];
    scores[String(colIdx)] = v;
    knownIds.add(bid);
    knownEntries.push({ id: bid, name: benchNameById(bid), value: v, source });
  }

  if (Object.keys(scores).length === 0) {
    errEl.textContent = 'Please enter at least one valid (benchmark, score) pair.';
    errEl.hidden = false;
    return;
  }

  const btn = document.getElementById('predict-btn');
  const allBtn = document.getElementById('predict-all-btn');
  btn.disabled = true;
  allBtn.disabled = true;
  const activeBtn = showAll ? allBtn : btn;
  activeBtn.textContent = 'Predicting… (~30s)';

  // Defer to next frame so the UI updates before Python runs
  await new Promise(r => setTimeout(r, 30));

  let preds;
  try {
    STATE.pyodide.globals.set('NEW_SCORES_JSON', JSON.stringify(scores));
    STATE.pyodide.runPython(`
import json
new_scores = json.loads(NEW_SCORES_JSON)
result_list = predictor.predict_new_model(M_OBS, new_scores)
del NEW_SCORES_JSON
`);
    preds = STATE.pyodide.globals.get('result_list').toJs();
  } catch (e) {
    console.error(e);
    errEl.textContent = 'Prediction failed: ' + e.message;
    errEl.hidden = false;
    btn.disabled = false;
    allBtn.disabled = false;
    btn.textContent = 'Predict benchmark';
    allBtn.textContent = 'All benchmarks';
    return;
  }

  renderResults(modelName, preds, scores, knownIds, knownEntries, targetBenchId, showAll);
  btn.disabled = false;
  allBtn.disabled = false;
  btn.textContent = 'Predict benchmark';
  allBtn.textContent = 'All benchmarks';
}

// -----------------------------------------------------------
// Results
// -----------------------------------------------------------
function renderResults(modelName, preds, scores, knownIds, knownEntries, targetBenchId, showAll) {
  document.getElementById('loader').hidden = true;
  document.getElementById('form').hidden = true;
  document.getElementById('results').hidden = false;
  document.getElementById('result-name').textContent = modelName;

  const halfWidths = STATE.data.meta?.prediction_interval?.benchmark_half_width || [];
  const trustProbabilities = STATE.data.meta?.prediction_interval?.benchmark_trust_probability || [];
  const rows = STATE.data.benchmarks.map((b, i) => ({
    id: b.id,
    name: b.name,
    val: knownIds.has(b.id) ? Number(scores[String(i)]) : preds[i],
    known: knownIds.has(b.id),
    trustProbability: trustProbabilities[i],
    interval: intervalForPrediction(preds[i], halfWidths[i], i),
  }));

  // Sort: known first, then predicted descending; but only if score is in [0,100] for sort
  rows.sort((a, b) => b.val - a.val);

  renderResultsTable(rows);
  renderEvidenceCard(knownEntries);
  renderSummaryCard(rows, targetBenchId, showAll);
  renderShareCard(rows, modelName, knownEntries, targetBenchId, showAll);
  renderAdviceCard(rows, scores, knownIds);

  document.getElementById('filter').oninput = (e) => {
    const q = e.target.value.toLowerCase();
    const filtered = rows.filter(r => r.name.toLowerCase().includes(q));
    renderResultsTable(filtered);
  };
  document.getElementById('predict-again').onclick = (e) => {
    e.preventDefault();
    document.getElementById('results').hidden = true;
    document.getElementById('form').hidden = false;
    document.getElementById('filter').value = '';
  };
  // Scroll to top so user sees results
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderShareCard(rows, modelName, knownEntries, targetBenchId, showAll) {
  const el = document.getElementById('share-card');
  const payload = buildSharePayload(rows, modelName, knownEntries, targetBenchId, showAll);
  el.hidden = false;

  const issueUrl = githubIssueUrl(payload);
  const emailUrl = emailShareUrl(payload);
  const xUrl = xShareUrl(payload);
  const facebookUrl = facebookShareUrl();

  el.innerHTML = `
    <div class="advice-kicker">Share / contribute</div>
    <h3>Keep the result or send public scores back.</h3>
    <p class="share-note">Nothing is uploaded automatically. These buttons only copy, download, or open a draft you can review.</p>
    <div class="share-actions">
      <button type="button" id="copy-summary" class="btn-primary">Copy summary</button>
      <button type="button" id="download-results" class="btn-secondary">Download JSON</button>
      <a class="btn-secondary" href="${issueUrl}" target="_blank" rel="noopener">Contribute scores</a>
      <a class="btn-secondary" href="${emailUrl}">Email us</a>
      <a class="btn-secondary" href="${xUrl}" target="_blank" rel="noopener">Share on X</a>
      <a class="btn-secondary" href="${facebookUrl}" target="_blank" rel="noopener">Facebook</a>
    </div>
  `;

  document.getElementById('copy-summary').onclick = () => copyShareSummary(payload);
  document.getElementById('download-results').onclick = () => downloadShareJson(payload);
}

function buildSharePayload(rows, modelName, knownEntries, targetBenchId, showAll) {
  const targetRow = targetBenchId ? rows.find(r => r.id === targetBenchId) : null;
  const predictions = showAll ? rows.filter(r => !r.known && isFinite(r.val)).slice(0, 8)
    : (targetRow ? [targetRow] : []);
  const observed = knownEntries.map(e => ({
    benchmark: e.name,
    score: Number(e.value),
    source: e.source || '',
  }));
  return {
    model: modelName,
    mode: showAll ? 'all benchmarks' : 'target benchmark',
    generated_with: 'BenchPress',
    url: window.location.origin + window.location.pathname,
    observed_scores: observed,
    predictions: predictions.map(r => ({
      benchmark: r.name,
      score: roundScore(r.val),
      kind: r.known ? 'provided score' : 'BenchPress prediction',
      trust: (!r.known && isFinite(r.trustProbability)) ? Math.round(100 * r.trustProbability) / 100 : null,
      interval_90: (!r.known && r.interval) ? r.interval.map(roundScore) : null,
    })),
  };
}

function shareSummaryText(payload) {
  const predictionLines = payload.predictions.map(p => {
    const trust = p.trust === null ? '' : ` (${Math.round(100 * p.trust)}% trust)`;
    return `- ${p.benchmark}: ${p.score}${trust}`;
  }).join('\n');
  const observedLines = payload.observed_scores.slice(0, 10).map(s => {
    const source = s.source ? ` — ${s.source}` : '';
    return `- ${s.benchmark}: ${roundScore(s.score)}${source}`;
  }).join('\n');
  return [
    `BenchPress prediction for ${payload.model}`,
    '',
    'Prediction:',
    predictionLines || '- No prediction selected',
    '',
    'Scores provided:',
    observedLines || '- None',
    '',
    payload.url,
  ].join('\n');
}

async function copyShareSummary(payload) {
  const text = shareSummaryText(payload);
  const btn = document.getElementById('copy-summary');
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
    btn.textContent = 'Copied';
    setTimeout(() => { btn.textContent = 'Copy summary'; }, 1600);
  } catch (e) {
    console.error(e);
    btn.textContent = 'Copy failed';
    setTimeout(() => { btn.textContent = 'Copy summary'; }, 1600);
  }
}

function downloadShareJson(payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${slugify(payload.model)}-benchpress-results.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function githubIssueUrl(payload) {
  const title = `Score contribution: ${payload.model}`;
  const body = [
    '### Model',
    payload.model,
    '',
    '### Public score sources',
    payload.observed_scores.map(s => `- ${s.benchmark}: ${roundScore(s.score)}${s.source ? ` — ${s.source}` : ' — source needed'}`).join('\n') || '- ',
    '',
    '### BenchPress output',
    payload.predictions.map(p => `- ${p.benchmark}: ${p.score}`).join('\n') || '- ',
  ].join('\n');
  return `https://github.com/microsoft/benchpress/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}

function emailShareUrl(payload) {
  const subject = `BenchPress result for ${payload.model}`;
  return `mailto:zengyuchen@microsoft.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(shareSummaryText(payload))}`;
}

function xShareUrl(payload) {
  const first = payload.predictions[0];
  const text = first
    ? `I predicted ${first.benchmark} for ${payload.model} with BenchPress: ${first.score}.`
    : `I tried BenchPress on ${payload.model}.`;
  return `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(payload.url)}`;
}

function facebookShareUrl() {
  return `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(window.location.origin + window.location.pathname)}`;
}

function roundScore(value) {
  return Number.isFinite(value) ? Math.round(value * 10) / 10 : null;
}

function slugify(value) {
  return String(value || 'model').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'model';
}

function renderSummaryCard(rows, targetBenchId, showAll) {
  const el = document.getElementById('summary-card');
  const targetRow = targetBenchId ? rows.find(r => r.id === targetBenchId) : null;
  const displayRows = showAll ? rows
    .filter(r => !r.known && isFinite(r.val))
    .slice(0, 8) : (targetRow ? [targetRow] : []);
  if (!displayRows.length) {
    el.innerHTML = '';
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const items = displayRows.map((r, idx) => {
    const width = Math.max(4, Math.min(100, r.val));
    const trust = (!r.known && isFinite(r.trustProbability))
      ? `${Math.round(100 * r.trustProbability)}% trust`
      : (r.known ? 'provided score' : 'trust unavailable');
    return `<li>
      <div class="summary-row-head">
        <span class="summary-rank">${showAll ? `#${idx + 1}` : 'target'}</span>
        <strong>${escapeHtml(r.name)}</strong>
        <span>${r.val.toFixed(1)}</span>
      </div>
      <div class="summary-bar"><span style="width:${width}%"></span></div>
      <p>${trust}${r.interval ? ` · ${r.interval[0].toFixed(1)}-${r.interval[1].toFixed(1)}` : ''}</p>
    </li>`;
  }).join('');
  el.innerHTML = `
    <div class="advice-kicker">${showAll ? 'Top predictions' : 'Target prediction'}</div>
    <h3>${showAll ? 'What to look at first.' : 'The benchmark you asked for.'}</h3>
    <ul class="summary-list">${items}</ul>
  `;
}

function renderAdviceCard(rows, scores, knownIds) {
  const bestNext = nextMissingProbes(BEST_PREDICTIVE_PROBES, knownIds, 4);
  const lowCostNext = nextMissingProbes(LOW_COST_PROBES, knownIds, 4);
  const lowConfidence = rows
    .filter(r => !r.known && isFinite(r.trustProbability))
    .sort((a, b) => (a.trustProbability - b.trustProbability) || (intervalWidth(b) - intervalWidth(a)))
    .slice(0, 4);
  const neighbors = closestPublicNeighbors(scores, 3);

  const block = (title, items, formatter) => {
    if (!items.length) return '';
    return `<div class="advice-block"><h4>${escapeHtml(title)}</h4><ul>${items.map(x => `<li>${formatter(x)}</li>`).join('')}</ul></div>`;
  };

  const html = [
    '<div class="advice-kicker">Tips</div>',
    '<h3>Make this estimate more concrete.</h3>',
    '<p>BenchPress is more reliable when you add informative observed scores. Start with missing probes, then validate low-trust predictions that matter for your decision.</p>',
    block('Best next probes', bestNext, b => `${escapeHtml(b.name)} <span class="advice-dim">adds high predictive coverage</span>`),
    block('Low-cost alternatives', lowCostNext, b => `${escapeHtml(b.name)} <span class="advice-dim">cheap candidate set</span>`),
    block('Validate these low-confidence predictions', lowConfidence, r => `${escapeHtml(r.name)} <span class="advice-dim">trust ${Math.round(100 * r.trustProbability)}%, range ${r.interval ? `${r.interval[0].toFixed(1)}-${r.interval[1].toFixed(1)}` : 'wide'}</span>`),
    block('Closest public neighbors', neighbors, n => `${escapeHtml(n.name)} <span class="advice-dim">${n.provider ? escapeHtml(n.provider) + ', ' : ''}${n.overlap} shared score${n.overlap === 1 ? '' : 's'}</span>`),
  ].filter(Boolean).join('');

  document.getElementById('advice-card').innerHTML = html;
}

function renderEvidenceCard(entries) {
  const el = document.getElementById('evidence-card');
  if (!entries.length) {
    el.innerHTML = '';
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const items = entries.map(e => {
    const source = formatProvidedSource(e.source);
    return `<li>
      <div>
        <strong>${escapeHtml(e.name)}</strong>
        <span>${Number(e.value).toFixed(1)}</span>
      </div>
      ${source ? `<p>${source}</p>` : '<p class="advice-dim">No source attached</p>'}
    </li>`;
  }).join('');
  el.innerHTML = `
    <div class="advice-kicker">Your inputs</div>
    <h3>Scores used for this prediction.</h3>
    <ul class="evidence-list">${items}</ul>
  `;
}

function formatProvidedSource(source) {
  if (!source) return '';
  if (/^https?:\/\//i.test(source)) {
    const safe = escapeHtml(source);
    return `<a href="${safe}" target="_blank" rel="noopener">Provided source ↗</a>`;
  }
  return `Source note: ${escapeHtml(source)}`;
}

function nextMissingProbes(probeIds, knownIds, limit) {
  return probeIds
    .filter(id => !knownIds.has(id))
    .map(id => ({ id, name: benchNameById(id) }))
    .filter(b => b.name !== b.id)
    .slice(0, limit);
}

function closestPublicNeighbors(scores, limit) {
  const known = Object.entries(scores).map(([j, v]) => [Number(j), Number(v)]);
  if (!known.length) return [];
  return STATE.data.models.map((m, i) => {
    let total = 0;
    let overlap = 0;
    for (const [j, v] of known) {
      const obs = STATE.data.observed[i][j];
      if (obs === null || !isFinite(obs)) continue;
      total += Math.abs(obs - v);
      overlap += 1;
    }
    if (!overlap) return null;
    return { name: m.name, provider: m.provider, overlap, distance: total / overlap };
  }).filter(Boolean)
    .sort((a, b) => (b.overlap - a.overlap) || (a.distance - b.distance))
    .slice(0, limit);
}

function intervalWidth(row) {
  return row.interval ? row.interval[1] - row.interval[0] : Number.POSITIVE_INFINITY;
}

function benchNameById(id) {
  const b = STATE.data?.benchmarks?.find(x => x.id === id);
  return b ? b.name : id;
}

function renderResultsTable(rows) {
  const body = document.getElementById('results-body');
  body.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    const cls = r.known ? 'score-known' : 'score-pred';
    const badge = r.known ? '<span class="badge badge-known">KNOWN</span>' : '';
    const valStr = isFinite(r.val) ? r.val.toFixed(1) : '—';
    const rangeStr = (!r.known && r.interval)
      ? `${r.interval[0].toFixed(1)}–${r.interval[1].toFixed(1)}`
      : '—';
    const trustStr = (!r.known && isFinite(r.trustProbability))
      ? `${Math.round(100 * r.trustProbability)}%`
      : '—';
    tr.innerHTML = `
      <td>${escapeHtml(r.name)}${badge}</td>
      <td class="score-cell ${cls}">${valStr}</td>
      <td class="range-cell">${trustStr}</td>
      <td class="range-cell">${rangeStr}</td>
    `;
    body.appendChild(tr);
  }
}

function intervalForPrediction(value, halfWidth, benchIdx) {
  if (!isFinite(value) || !isFinite(halfWidth)) return null;
  let lo = value - halfWidth;
  let hi = value + halfWidth;
  if (isPercentLikeBenchmark(benchIdx)) {
    lo = Math.max(0, lo);
    hi = Math.min(100, hi);
  }
  return [lo, hi];
}

function isPercentLikeBenchmark(benchIdx) {
  const vals = STATE.data.observed
    .map(row => row[benchIdx])
    .filter(v => v !== null && isFinite(v));
  return vals.length > 0 && Math.min(...vals) >= -1 && Math.max(...vals) <= 101;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// -----------------------------------------------------------
// Go!
// -----------------------------------------------------------
boot();
