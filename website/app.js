(async () => {
  let data = null;
  let modelsSorted = [];
  let benchmarksSorted = [];

  const modelInput = document.getElementById('model');
  const benchInput = document.getElementById('bench');
  const modelList = document.getElementById('model-list');
  const benchList = document.getElementById('bench-list');
  const modelByLabel = new Map();
  const benchByLabel = new Map();
  const bigval = document.getElementById('bigval');
  const bigsub = document.getElementById('bigsub');
  const bigEyebrow = document.getElementById('big-eyebrow');
  const big = document.getElementById('big');
  const toprows = document.getElementById('toprows');
  const matrixSource = document.getElementById('matrix-source');
  const matrixStatus = document.getElementById('matrix-status');
  const matrixUpload = document.getElementById('matrix-upload');
  const matrixFiles = document.getElementById('matrix-files');
  const matrixRun = document.getElementById('matrix-run');
  const matrixDownload = document.getElementById('matrix-download');

  const modelLabel = model => `${model.name} — ${model.provider}`;
  const benchLabel = benchmark => benchmark.name;
  let currentI = 0;
  let currentJ = 0;
  let lbMode = 'full';
  let expandedI = null;

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    })[char]);
  }

  function setMatrixStatus(message, isError = false) {
    matrixStatus.textContent = message;
    matrixStatus.classList.toggle('error', isError);
  }

  function matrixSummary(matrix) {
    const name = matrix.meta?.name || 'Score matrix';
    const models = matrix.models.length;
    const benchmarks = matrix.benchmarks.length;
    const observed = matrix.meta?.observed_cells ?? matrix.observed.reduce(
      (total, row) => total + row.filter(Number.isFinite).length, 0);
    const confidence = matrix.meta?.confidence_available
      ? ''
      : ' · point predictions only; intervals are not calibrated for this matrix';
    return `${name}: ${models} models × ${benchmarks} benchmarks · ` +
      `${observed.toLocaleString()} observed scores${confidence}`;
  }

  function installMatrix(matrix) {
    data = matrix;
    modelsSorted = data.models.map((model, i) => ({...model, idx: i}))
      .sort((a, b) => a.name.localeCompare(b.name));
    benchmarksSorted = data.benchmarks.map((benchmark, j) => ({
      ...benchmark, idx: j,
    })).sort((a, b) => a.name.localeCompare(b.name));

    modelByLabel.clear();
    benchByLabel.clear();
    modelList.replaceChildren();
    benchList.replaceChildren();
    for (const model of modelsSorted) {
      const option = document.createElement('option');
      option.value = modelLabel(model);
      modelList.appendChild(option);
      modelByLabel.set(option.value, model.idx);
    }
    for (const benchmark of benchmarksSorted) {
      const option = document.createElement('option');
      option.value = benchLabel(benchmark);
      benchList.appendChild(option);
      benchByLabel.set(option.value, benchmark.idx);
    }

    const defaultModel = modelsSorted.find(
      model => model.name.toLowerCase().includes('gpt-5')) || modelsSorted[0];
    const defaultBenchmark = benchmarksSorted.find(
      benchmark => benchmark.name.toLowerCase().includes('gpqa')) ||
      benchmarksSorted[0];
    currentI = defaultModel.idx;
    currentJ = defaultBenchmark.idx;
    expandedI = null;
    modelInput.value = modelLabel(defaultModel);
    benchInput.value = benchLabel(defaultBenchmark);
    matrixDownload.hidden = !data.predictions || data.meta?.id !== 'upload';
    setMatrixStatus(matrixSummary(data));
    update();
  }

  async function loadMatrix(source) {
    matrixUpload.hidden = source !== 'upload';
    matrixRun.hidden = true;
    matrixDownload.hidden = true;
    if (source === 'upload') {
      setMatrixStatus(
        'Select scores.csv and optional scores.meta.json. Download the template if you need the exact format.');
      return;
    }
    setMatrixStatus(`Loading ${source === 'benchpress' ? 'BenchPress' : source.toUpperCase()}…`);
    try {
      installMatrix(await BenchPressMatrixSource.load(source));
    } catch (error) {
      console.error(error);
      setMatrixStatus(error.message, true);
    }
  }

  function formatInterval(interval) {
    if (!interval || interval.length !== 2 ||
        interval[0] === null || interval[1] === null) return '—';
    return `${interval[0].toFixed(1)}–${interval[1].toFixed(1)}`;
  }

  function formatTrustProbability(probability) {
    return Number.isFinite(probability) ?
      `${Math.round(100 * probability)}%` : '—';
  }

  function trustLine(probability) {
    return Number.isFinite(probability)
      ? `<br><span class="interval-text">Trust probability: ` +
        `${formatTrustProbability(probability)}</span>`
      : '';
  }

  function intervalLine(i, j) {
    const interval = data.prediction_intervals?.[i]?.[j];
    if (!interval) return '';
    return `<div class="meta-line interval-text">90% predicted range: ` +
      `${formatInterval(interval)}</div>`;
  }

  function settingLines(setting) {
    if (!setting) return '';
    const order = [
      'mode', 'effort', 'tools', 'sampling', 'judge', 'harness',
      'prompt_style', 'temperature', 'context', 'notes',
    ];
    return order.filter(key =>
      setting[key] !== undefined && setting[key] !== null &&
      setting[key] !== '').map(key =>
      `<div class="kv"><span class="k">${escapeHtml(key)}</span>` +
      `<span class="v">${escapeHtml(setting[key])}</span></div>`).join('');
  }

  function rowDetails(row) {
    const model = data.models[row.i];
    const parts = [];
    const reason = model.reasoning === true ? 'reasoning model' :
      (model.reasoning === false ? 'non-reasoning model' :
        escapeHtml(model.provider || 'model'));
    const release = model.release_date ?
      ` · released ${escapeHtml(model.release_date)}` : '';
    const weights = model.open_weights === true ? ' · open weights' :
      (model.open_weights === false ? ' · closed weights' : '');
    parts.push(`<div class="meta-line">${reason}${release}${weights}</div>`);

    if (row.obs !== null) {
      const source = data.sources?.[row.i]?.[currentJ];
      if (source) {
        const link = source.url
          ? `<div class="meta-line"><a href="${escapeHtml(source.url)}" ` +
            `target="_blank" rel="noopener">↗ source link</a>` +
            `${source.matches_canonical === false ? ' · non-canonical setting' : ''}` +
            `${source.audit_status ? ' · audit: ' + escapeHtml(source.audit_status) : ''}</div>`
          : '';
        parts.push(link);
        if (source.reported_setting) {
          parts.push(`<div class="kv-block"><div class="kv-title">` +
            `Reported setting</div>${settingLines(source.reported_setting)}</div>`);
        }
        if (source.notes) {
          parts.push(`<div class="notes">${escapeHtml(source.notes)}</div>`);
        }
      } else {
        parts.push(`<div class="meta-line dim">Source metadata unavailable.</div>`);
      }
    } else {
      parts.push(`<div class="meta-line dim">BenchPress prediction ` +
        `(Logit Bias ALS, rank=2, λ=0.1). No public score reported.</div>`);
      const trustProbability = data.trust_probabilities?.[row.i]?.[currentJ];
      if (Number.isFinite(trustProbability)) {
        parts.push(`<div class="meta-line interval-text">Trust probability: ` +
          `${formatTrustProbability(trustProbability)}</div>`);
      }
      parts.push(intervalLine(row.i, currentJ));
      if (model.canonical_setting) {
        parts.push(`<div class="kv-block"><div class="kv-title">` +
          `Model canonical setting (used as covariate)</div>` +
          `${settingLines(model.canonical_setting)}</div>`);
      }
    }
    return parts.join('');
  }

  function update() {
    if (!data) return;
    const model = data.models[currentI];
    const benchmark = data.benchmarks[currentJ];
    const predicted = data.predictions?.[currentI]?.[currentJ];
    const observed = data.observed[currentI][currentJ];
    const interval = data.prediction_intervals?.[currentI]?.[currentJ];
    const trustProbability = data.trust_probabilities?.[currentI]?.[currentJ];

    bigval.classList.remove('observed');
    big.querySelector('.bigtag')?.remove();
    if (observed !== null) {
      bigEyebrow.textContent = 'Reported score';
      bigval.textContent = observed.toFixed(1);
      bigval.classList.add('observed');
      bigsub.innerHTML = `<strong>${escapeHtml(model.name)}</strong> on ` +
        `${escapeHtml(benchmark.name)}`;
      const tag = document.createElement('div');
      tag.className = 'bigtag observed';
      tag.textContent = 'Reported';
      bigsub.after(tag);
    } else if (Number.isFinite(predicted)) {
      bigEyebrow.textContent = data.meta?.confidence_available
        ? 'Predicted score + trust' : 'Predicted score';
      bigval.textContent = predicted.toFixed(1);
      bigsub.innerHTML = `<strong>${escapeHtml(model.name)}</strong> on ` +
        `${escapeHtml(benchmark.name)}<br>No public score reported` +
        `${trustLine(trustProbability)}` +
        `${interval ? `<br><span class="interval-text">90% predicted range: ` +
          `${formatInterval(interval)}</span>` : ''}`;
      const tag = document.createElement('div');
      tag.className = 'bigtag predicted';
      tag.textContent = 'Predicted';
      bigsub.after(tag);
    } else {
      bigEyebrow.textContent = 'Score';
      bigval.textContent = '—';
      bigsub.textContent = data.meta?.id === 'upload'
        ? 'Select Run BenchPress above to fill missing scores.'
        : 'No prediction available.';
    }
    renderTop(currentJ, currentI);
  }

  function renderTop(j, selectedI) {
    const benchmark = data.benchmarks[j];
    document.getElementById('lb-title').textContent = `On ${benchmark.name}.`;
    const all = data.models.map((model, i) => ({
      i,
      name: model.name,
      provider: model.provider,
      pred: data.predictions?.[i]?.[j] ?? null,
      obs: data.observed[i][j],
    }));
    const scoreRange = benchmark.metric?.range;
    let rows = all.filter(row => {
      if (lbMode === 'obs') return row.obs !== null;
      const value = row.obs ?? row.pred;
      if (!Number.isFinite(value)) return false;
      return !scoreRange || (value >= scoreRange[0] && value <= scoreRange[1]);
    });
    rows.sort((a, b) => {
      const aScore = a.obs ?? a.pred;
      const bScore = b.obs ?? b.pred;
      return benchmark.metric?.higher_is_better === false
        ? aScore - bScore
        : bScore - aScore;
    });
    const rankIndex = rows.findIndex(row => row.i === selectedI);
    const rank = rankIndex >= 0 ? rankIndex + 1 : null;
    const rankElement = document.getElementById('lb-myrank');
    const selectedName = data.models[selectedI].name;
    rankElement.innerHTML = rank
      ? `<strong>${escapeHtml(selectedName)}</strong> ranks ` +
        `<strong>#${rank}</strong> of ${rows.length}.`
      : `<strong>${escapeHtml(selectedName)}</strong> has no ` +
        `${lbMode === 'obs' ? 'reported ' : ''}score on this benchmark.`;
    toprows.innerHTML = rows.map((row, k) => {
      const isObserved = row.obs !== null;
      const value = isObserved ? row.obs : row.pred;
      const className = isObserved ? 'observed' : 'predicted';
      const highlighted = row.i === selectedI ? ' rowitem-current' : '';
      const expanded = row.i === expandedI ? ' rowitem-expanded' : '';
      const interval = data.prediction_intervals?.[row.i]?.[j];
      const intervalHtml = !isObserved && interval
        ? `<div class="val-interval">${formatInterval(interval)}</div>` : '';
      const trustProbability = data.trust_probabilities?.[row.i]?.[j];
      const trustHtml = !isObserved && Number.isFinite(trustProbability)
        ? `<div class="val-interval">trust ` +
          `${formatTrustProbability(trustProbability)}</div>` : '';
      return `<div class="rowitem${highlighted}${expanded}" data-i="${row.i}">
        <div class="rowmain">
          <div class="rank">#${k + 1}</div>
          <div class="name">${escapeHtml(row.name)}</div>
          <div class="tag ${className}">${isObserved ? 'reported' : 'predicted'}</div>
          <div class="valwrap"><div class="val ${className}">${value.toFixed(1)}</div>${trustHtml}${intervalHtml}</div>
          <div class="chev">▾</div>
        </div>
        <div class="rowdetails">${row.i === expandedI ? rowDetails(row) : ''}</div>
      </div>`;
    }).join('');
  }

  function makeCombo(input, byLabel, getLabel, getCurrent, setCurrent) {
    function commit() {
      const value = input.value.trim();
      if (byLabel.has(value)) {
        setCurrent(byLabel.get(value));
        update();
      }
    }
    input.addEventListener('input', commit);
    input.addEventListener('change', commit);
    input.addEventListener('focus', () => {
      input.dataset.prev = input.value;
      input.value = '';
    });
    input.addEventListener('blur', () => {
      const value = input.value.trim();
      if (!byLabel.has(value) && data) {
        input.value = input.dataset.prev || getLabel(getCurrent());
      }
    });
  }

  function csvEscape(value) {
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function downloadPredictions() {
    if (!data?.predictions) return;
    const rowSupport = data.observed.map(
      row => row.filter(Number.isFinite).length);
    const columnSupport = data.benchmarks.map((_, j) =>
      data.observed.reduce(
        (count, row) => count + Number(Number.isFinite(row[j])), 0));
    const rows = [[
      'model', 'benchmark', 'score', 'status',
      'row_support', 'column_support', 'source_matrix',
    ]];
    data.models.forEach((model, i) => data.benchmarks.forEach((benchmark, j) => {
      if (Number.isFinite(data.observed[i][j])) return;
      const score = data.predictions[i][j];
      if (!Number.isFinite(score)) return;
      rows.push([
        model.id, benchmark.id, score.toFixed(6), 'predicted',
        rowSupport[i], columnSupport[j], data.meta.id,
      ]);
    }));
    const csv = rows.map(
      row => row.map(csvEscape).join(',')).join('\n') + '\n';
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
    link.download = 'benchpress-predictions.csv';
    link.click();
    URL.revokeObjectURL(link.href);
  }

  toprows.addEventListener('click', event => {
    const row = event.target.closest('.rowitem');
    if (!row) return;
    const i = Number(row.dataset.i);
    expandedI = expandedI === i ? null : i;
    renderTop(currentJ, currentI);
  });

  document.querySelectorAll('.lb-mode').forEach(element => {
    element.addEventListener('click', event => {
      event.preventDefault();
      lbMode = element.dataset.mode;
      document.querySelectorAll('.lb-mode').forEach(candidate =>
        candidate.classList.toggle('active', candidate === element));
      update();
    });
  });

  makeCombo(
    modelInput,
    modelByLabel,
    index => modelLabel(data.models[index]),
    () => currentI,
    index => { currentI = index; },
  );
  makeCombo(
    benchInput,
    benchByLabel,
    index => benchLabel(data.benchmarks[index]),
    () => currentJ,
    index => { currentJ = index; },
  );

  matrixSource.addEventListener('change', () => loadMatrix(matrixSource.value));
  matrixFiles.addEventListener('change', async () => {
    matrixRun.hidden = true;
    matrixDownload.hidden = true;
    try {
      const matrix = await BenchPressMatrixSource.readUpload(matrixFiles.files);
      installMatrix(matrix);
      matrixRun.hidden = false;
      const warningText = matrix.warnings.length
        ? ` Valid with support warnings: ${matrix.warnings.join(' ')}`
        : ' Matrix is valid.';
      setMatrixStatus(matrixSummary(matrix) + warningText);
    } catch (error) {
      console.error(error);
      setMatrixStatus(error.message, true);
    }
  });
  matrixRun.addEventListener('click', async () => {
    matrixRun.disabled = true;
    setMatrixStatus('Loading the local predictor and filling missing cells…');
    try {
      installMatrix(await BenchPressMatrixSource.complete(data));
      matrixRun.hidden = true;
      matrixDownload.hidden = false;
      setMatrixStatus(matrixSummary(data) + ' · prediction complete');
    } catch (error) {
      console.error(error);
      setMatrixStatus(`Prediction failed: ${error.message}`, true);
    } finally {
      matrixRun.disabled = false;
    }
  });
  matrixDownload.addEventListener('click', downloadPredictions);

  await loadMatrix('benchpress');
})();
