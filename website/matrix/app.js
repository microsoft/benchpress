const STATE = {
  matrix: null,
  pyodide: null,
  predictorReady: false,
  source: 'benchpress',
};

const SOURCE_PATHS = {
  eee: 'matrices/eee.json.gz?v=1',
  helm: 'matrices/helm.json.gz?v=1',
};

function setStatus(message, isError = false) {
  const status = document.getElementById('status');
  status.textContent = message;
  status.classList.toggle('error', isError);
}

function csvCells(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      row.push(cell);
      cell = '';
    } else if (char === '\n') {
      row.push(cell.replace(/\r$/, ''));
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += char;
    }
  }
  if (quoted) throw new Error('CSV contains an unclosed quoted field.');
  if (cell || row.length) {
    row.push(cell.replace(/\r$/, ''));
    rows.push(row);
  }
  return rows.filter(values => values.some(value => value.trim()));
}

function validateUploadedMatrix(csvText, metadata) {
  const rows = csvCells(csvText);
  if (!rows.length || rows[0][0].replace(/^\uFEFF/, '').trim() !== 'model') {
    throw new Error("The first CSV header cell must be exactly 'model'.");
  }
  const benchmarkIds = rows[0].slice(1).map(value => value.trim());
  if (benchmarkIds.length < 3 || benchmarkIds.some(value => !value)) {
    throw new Error('The matrix needs at least 3 non-empty benchmark ids.');
  }
  if (new Set(benchmarkIds).size !== benchmarkIds.length) {
    throw new Error('Benchmark ids must be unique.');
  }
  if (rows.length - 1 < 3) {
    throw new Error('The matrix needs at least 3 models.');
  }
  if (rows.length > 1001 || benchmarkIds.length > 1000) {
    throw new Error('The browser limit is 1,000 models by 1,000 benchmarks.');
  }

  const unknownMetadata = Object.keys(metadata).filter(
    benchmarkId => !benchmarkIds.includes(benchmarkId));
  if (unknownMetadata.length) {
    throw new Error(`Metadata includes unknown benchmark: ${unknownMetadata[0]}`);
  }

  const modelIds = [];
  const observed = [];
  for (let line = 1; line < rows.length; line += 1) {
    const values = rows[line];
    if (values.length > benchmarkIds.length + 1) {
      throw new Error(`CSV row ${line + 1} has too many cells.`);
    }
    while (values.length < benchmarkIds.length + 1) values.push('');
    const modelId = values[0].trim();
    if (!modelId) throw new Error(`CSV row ${line + 1} has an empty model id.`);
    modelIds.push(modelId);
    observed.push(values.slice(1).map((value, j) => {
      const text = value.trim();
      if (!text) return null;
      const score = Number(text.replace(/%$/, ''));
      if (!Number.isFinite(score)) {
        throw new Error(
          `Score for ${modelId}/${benchmarkIds[j]} must be numeric or blank.`);
      }
      return score;
    }));
  }
  if (new Set(modelIds).size !== modelIds.length) {
    throw new Error('Model ids must be unique.');
  }

  const metrics = benchmarkIds.map(benchmarkId => {
    const spec = metadata[benchmarkId] || {
      type: 'pct', range: [0, 100], higher_is_better: true,
    };
    if (!spec || typeof spec !== 'object' || !spec.type ||
        !Array.isArray(spec.range) || spec.range.length !== 2) {
      throw new Error(
        `Metadata for ${benchmarkId} needs type and range: [min, max].`);
    }
    const range = spec.range.map(Number);
    if (!range.every(Number.isFinite) || range[0] > range[1]) {
      throw new Error(`Metadata range for ${benchmarkId} is invalid.`);
    }
    return {
      type: String(spec.type).toLowerCase(),
      range,
      higher_is_better: spec.higher_is_better !== false,
    };
  });

  let observedCells = 0;
  const rowSupport = observed.map(row => row.filter(Number.isFinite).length);
  const columnSupport = benchmarkIds.map((_, j) =>
    observed.reduce((count, row) => count + Number(Number.isFinite(row[j])), 0));
  observed.forEach((row, i) => row.forEach((score, j) => {
    if (!Number.isFinite(score)) return;
    observedCells += 1;
    const [lower, upper] = metrics[j].range;
    if (score < lower || score > upper) {
      throw new Error(
        `${modelIds[i]}/${benchmarkIds[j]} is outside [${lower}, ${upper}].`);
    }
  }));
  if (observedCells < 3) throw new Error('The matrix needs at least 3 scores.');
  if (observedCells > 100000) {
    throw new Error('The browser limit is 100,000 observed scores.');
  }
  if (rowSupport.some(count => count === 0)) {
    throw new Error('Every model needs at least one observed score.');
  }
  if (columnSupport.some(count => count === 0)) {
    throw new Error('Every benchmark needs at least one observed score.');
  }

  const lowRows = rowSupport.filter(count => count < 15).length;
  const lowColumns = columnSupport.filter(count => count < 8).length;
  const warnings = [];
  if (lowRows) warnings.push(
    `${lowRows} models have fewer than 15 observed benchmarks.`);
  if (lowColumns) warnings.push(
    `${lowColumns} benchmarks have fewer than 8 observed models.`);
  return {
    meta: {
      id: 'upload',
      name: 'Uploaded matrix',
      models: modelIds.length,
      benchmarks: benchmarkIds.length,
      observed_cells: observedCells,
      confidence_available: false,
    },
    models: modelIds.map(id => ({id, name: id, provider: 'Uploaded'})),
    benchmarks: benchmarkIds.map((id, j) => ({
      id, name: id, metric: metrics[j],
    })),
    observed,
    predictions: null,
    warnings,
  };
}

async function loadGzipJson(url) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser cannot open compressed matrix artifacts.');
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load matrix (${response.status}).`);
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text());
}

async function loadSource(source) {
  STATE.source = source;
  document.getElementById('upload-panel').hidden = source !== 'upload';
  document.getElementById('workspace').hidden = true;
  document.getElementById('custom-run').hidden = true;
  document.getElementById('validation').hidden = true;
  if (source === 'upload') {
    STATE.matrix = null;
    setStatus('Select scores.csv and optional scores.meta.json.');
    return;
  }
  setStatus(`Loading ${source === 'benchpress' ? 'BenchPress' : source.toUpperCase()}…`);
  try {
    if (source === 'benchpress') {
      const response = await fetch('../data.json?v=7');
      if (!response.ok) throw new Error(`Could not load matrix (${response.status}).`);
      STATE.matrix = await response.json();
      STATE.matrix.meta.id = 'benchpress';
      STATE.matrix.meta.name = 'BenchPress curated snapshot';
      STATE.matrix.meta.models = STATE.matrix.models.length;
      STATE.matrix.meta.benchmarks = STATE.matrix.benchmarks.length;
      STATE.matrix.meta.observed_cells = STATE.matrix.observed.reduce(
        (total, row) => total + row.filter(Number.isFinite).length, 0);
      STATE.matrix.meta.confidence_available = true;
      STATE.matrix.benchmarks.forEach(benchmark => {
        benchmark.metric = {type: 'pct', range: [0, 100],
          higher_is_better: true};
      });
    } else {
      STATE.matrix = await loadGzipJson(SOURCE_PATHS[source]);
    }
    showMatrix();
  } catch (error) {
    console.error(error);
    setStatus(error.message, true);
  }
}

function showMatrix() {
  const matrix = STATE.matrix;
  const possible = matrix.meta.models * matrix.meta.benchmarks;
  const fill = possible ? 100 * matrix.meta.observed_cells / possible : 0;
  document.getElementById('matrix-summary').innerHTML = `
    <h3>${escapeHtml(matrix.meta.name)}</h3>
    <p>${sourceDescription(matrix.meta)}</p>
    <div class="summary-stats">
      <div class="summary-stat"><strong>${matrix.meta.models}</strong><span>models</span></div>
      <div class="summary-stat"><strong>${matrix.meta.benchmarks}</strong><span>benchmarks</span></div>
      <div class="summary-stat"><strong>${matrix.meta.observed_cells.toLocaleString()}</strong><span>observed scores</span></div>
      <div class="summary-stat"><strong>${fill.toFixed(1)}%</strong><span>filled</span></div>
    </div>`;
  const modelSelect = document.getElementById('model-select');
  const benchmarkSelect = document.getElementById('benchmark-select');
  modelSelect.replaceChildren(...matrix.models.map((model, i) =>
    optionElement(i, model.name)));
  benchmarkSelect.replaceChildren(...matrix.benchmarks.map((benchmark, j) =>
    optionElement(j, benchmark.name)));
  modelSelect.onchange = renderSelection;
  benchmarkSelect.onchange = renderSelection;
  document.getElementById('filter').oninput = renderModelTable;
  document.getElementById('download-predictions').onclick = downloadPredictions;
  document.getElementById('workspace').hidden = false;
  document.getElementById('download-predictions').disabled = !matrix.predictions;
  setStatus(`${matrix.meta.name} is ready.`);
  renderSelection();
}

function sourceDescription(meta) {
  if (meta.id === 'upload') return 'Validated locally. No uploaded data leaves this browser.';
  if (meta.id === 'benchpress') return 'Paper-canonical curated score matrix with BenchPress-calibrated trust estimates.';
  return `Source snapshot ${meta.snapshot_date}; percentage benchmarks filtered to at least 15 scores per model and 8 models per benchmark. External matrices show point predictions and support, not BenchPress-calibrated intervals.`;
}

function optionElement(value, label) {
  const option = document.createElement('option');
  option.value = String(value);
  option.textContent = label;
  return option;
}

function supportCounts(matrix) {
  return {
    rows: matrix.observed.map(row => row.filter(Number.isFinite).length),
    columns: matrix.benchmarks.map((_, j) =>
      matrix.observed.reduce(
        (count, row) => count + Number(Number.isFinite(row[j])), 0)),
  };
}

function renderSelection() {
  const matrix = STATE.matrix;
  const i = Number(document.getElementById('model-select').value);
  const j = Number(document.getElementById('benchmark-select').value);
  const observed = matrix.observed[i][j];
  const predicted = matrix.predictions?.[i]?.[j];
  const isObserved = Number.isFinite(observed);
  const score = isObserved ? observed : predicted;
  const supports = supportCounts(matrix);
  const result = document.getElementById('cell-result');
  result.replaceChildren();
  const title = document.createElement('h3');
  title.textContent = `${matrix.models[i].name} on ${matrix.benchmarks[j].name}`;
  const value = document.createElement('div');
  value.className = 'cell-value';
  value.textContent = Number.isFinite(score) ? score.toFixed(1) : '—';
  const description = document.createElement('p');
  description.textContent = isObserved ? 'Reported score' :
    (Number.isFinite(score) ? 'BenchPress prediction' :
      'Run BenchPress to predict this missing score.');
  const metadata = document.createElement('div');
  metadata.className = 'cell-meta';
  [
    `row support ${supports.rows[i]}`,
    `benchmark support ${supports.columns[j]}`,
    isObserved ? 'observed' : 'predicted',
  ].forEach(text => {
    const span = document.createElement('span');
    span.textContent = text;
    metadata.appendChild(span);
  });
  if (!isObserved && matrix.meta.confidence_available) {
    const trust = matrix.trust_probabilities?.[i]?.[j];
    const interval = matrix.prediction_intervals?.[i]?.[j];
    if (Number.isFinite(trust)) {
      const span = document.createElement('span');
      span.textContent = `trust ${Math.round(100 * trust)}%`;
      metadata.appendChild(span);
    }
    if (interval) {
      const span = document.createElement('span');
      span.textContent = `90% range ${interval[0].toFixed(1)}–${interval[1].toFixed(1)}`;
      metadata.appendChild(span);
    }
  }
  result.append(title, value, description, metadata);
  renderModelTable();
}

function renderModelTable() {
  if (!STATE.matrix) return;
  const matrix = STATE.matrix;
  const i = Number(document.getElementById('model-select').value);
  const query = document.getElementById('filter').value.trim().toLowerCase();
  const supports = supportCounts(matrix);
  const body = document.getElementById('results-body');
  body.replaceChildren();
  matrix.benchmarks.forEach((benchmark, j) => {
    if (query && !benchmark.name.toLowerCase().includes(query)) return;
    const observed = matrix.observed[i][j];
    const predicted = matrix.predictions?.[i]?.[j];
    const isObserved = Number.isFinite(observed);
    const score = isObserved ? observed : predicted;
    const tr = document.createElement('tr');
    [
      benchmark.name,
      Number.isFinite(score) ? score.toFixed(2) : '—',
      isObserved ? 'observed' : (Number.isFinite(score) ? 'predicted' : 'pending'),
      `row ${supports.rows[i]}, col ${supports.columns[j]}`,
    ].forEach(text => {
      const td = document.createElement('td');
      td.textContent = text;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

async function loadPredictor() {
  if (STATE.predictorReady) return;
  setStatus('Loading the local prediction runtime…');
  STATE.pyodide = await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
  });
  await STATE.pyodide.loadPackage(['numpy']);
  const response = await fetch('../add-model/predictor.py?v=4');
  const source = await response.text();
  STATE.pyodide.FS.writeFile('predictor.py', source);
  STATE.pyodide.runPython('import predictor');
  STATE.predictorReady = true;
}

async function runCustomPrediction() {
  const button = document.getElementById('run-prediction');
  const progress = document.getElementById('runtime-progress');
  button.disabled = true;
  progress.hidden = false;
  try {
    await loadPredictor();
    STATE.pyodide.globals.set('MATRIX_JSON', JSON.stringify(STATE.matrix.observed));
    STATE.pyodide.globals.set('METRIC_JSON', JSON.stringify(
      STATE.matrix.benchmarks.map(benchmark => benchmark.metric.type)));
    STATE.pyodide.runPython(`
import json
matrix_values = json.loads(MATRIX_JSON)
metric_types = json.loads(METRIC_JSON)
completed_matrix = predictor.predict_matrix(matrix_values, metric_types)
del MATRIX_JSON, METRIC_JSON
`);
    STATE.matrix.predictions = STATE.pyodide.globals.get(
      'completed_matrix').toJs();
    document.getElementById('download-predictions').disabled = false;
    setStatus('Prediction complete. Results remain in this browser.');
    renderSelection();
  } catch (error) {
    console.error(error);
    setStatus(`Prediction failed: ${error.message}`, true);
  } finally {
    button.disabled = false;
    progress.hidden = true;
  }
}

async function handleUpload(files) {
  const csvFile = [...files].find(file => file.name.toLowerCase().endsWith('.csv'));
  const metaFile = [...files].find(
    file => file.name.toLowerCase().endsWith('.meta.json'));
  if (!csvFile) {
    setStatus('Select one scores.csv file.', true);
    return;
  }
  if (csvFile.size > 20 * 1024 * 1024 ||
      (metaFile && metaFile.size > 20 * 1024 * 1024)) {
    setStatus('Each uploaded file must be 20 MB or smaller.', true);
    return;
  }
  try {
    const metadata = metaFile ? JSON.parse(await metaFile.text()) : {};
    if (!metadata || Array.isArray(metadata) || typeof metadata !== 'object') {
      throw new Error('scores.meta.json must contain a JSON object.');
    }
    STATE.matrix = validateUploadedMatrix(await csvFile.text(), metadata);
    showMatrix();
    const validation = document.getElementById('validation');
    validation.hidden = false;
    validation.innerHTML = STATE.matrix.warnings.length
      ? `<strong>Valid with support warnings</strong><ul>${STATE.matrix.warnings.map(
          warning => `<li>${escapeHtml(warning)}</li>`).join('')}</ul>`
      : '<strong>Valid matrix</strong><p>All rows and columns meet the recommended support thresholds.</p>';
    document.getElementById('custom-run').hidden = false;
    document.getElementById('run-prediction').onclick = runCustomPrediction;
  } catch (error) {
    console.error(error);
    STATE.matrix = null;
    document.getElementById('workspace').hidden = true;
    setStatus(error.message, true);
  }
}

function csvEscape(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadPredictions() {
  const matrix = STATE.matrix;
  const supports = supportCounts(matrix);
  const rows = [[
    'model', 'benchmark', 'score', 'status',
    'row_support', 'column_support', 'source_matrix',
  ]];
  matrix.models.forEach((model, i) => matrix.benchmarks.forEach((benchmark, j) => {
    if (Number.isFinite(matrix.observed[i][j])) return;
    const score = matrix.predictions?.[i]?.[j];
    if (!Number.isFinite(score)) return;
    rows.push([
      model.id, benchmark.id, score.toFixed(6), 'predicted',
      supports.rows[i], supports.columns[j], matrix.meta.id,
    ]);
  }));
  const csv = rows.map(row => row.map(csvEscape).join(',')).join('\n') + '\n';
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  link.download = `${matrix.meta.id}-benchpress-predictions.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

document.querySelectorAll('.source-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.source-card').forEach(
      candidate => candidate.classList.toggle('active', candidate === card));
    loadSource(card.dataset.source);
  });
});
document.getElementById('matrix-files').addEventListener(
  'change', event => handleUpload(event.target.files));
loadSource('benchpress');
