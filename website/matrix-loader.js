const BenchPressMatrixSource = (() => {
  let pyodide = null;
  let predictorReady = false;

  function csvRows(text) {
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

  function validateUpload(csvText, metadata) {
    const rows = csvRows(csvText);
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
      if (spec.higher_is_better !== undefined &&
          typeof spec.higher_is_better !== 'boolean') {
        throw new Error(
          `Metadata higher_is_better for ${benchmarkId} must be true or false.`);
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

    const warnings = [];
    const lowRows = rowSupport.filter(count => count < 15).length;
    const lowColumns = columnSupport.filter(count => count < 8).length;
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

  async function load(source) {
    if (source === 'benchpress') {
      const response = await fetch('data.json?v=7');
      if (!response.ok) throw new Error(`Could not load matrix (${response.status}).`);
      const data = await response.json();
      data.meta.id = 'benchpress';
      data.meta.name = 'BenchPress curated snapshot';
      data.meta.models = data.models.length;
      data.meta.benchmarks = data.benchmarks.length;
      data.meta.observed_cells = data.observed.reduce(
        (total, row) => total + row.filter(Number.isFinite).length, 0);
      data.meta.confidence_available = true;
      data.benchmarks.forEach(benchmark => {
        benchmark.metric = {
          type: 'pct', range: [0, 100], higher_is_better: true,
        };
      });
      return data;
    }
    return loadGzipJson(`matrix/matrices/${source}.json.gz?v=1`);
  }

  async function readUpload(files) {
    const selected = [...files];
    const csvFile = selected.find(file => file.name.toLowerCase().endsWith('.csv'));
    const metadataFile = selected.find(
      file => file.name.toLowerCase().endsWith('.meta.json'));
    if (!csvFile) throw new Error('Select one scores.csv file.');
    if (csvFile.size > 20 * 1024 * 1024 ||
        (metadataFile && metadataFile.size > 20 * 1024 * 1024)) {
      throw new Error('Each uploaded file must be 20 MB or smaller.');
    }
    const metadata = metadataFile ? JSON.parse(await metadataFile.text()) : {};
    if (!metadata || Array.isArray(metadata) || typeof metadata !== 'object') {
      throw new Error('scores.meta.json must contain a JSON object.');
    }
    return validateUpload(await csvFile.text(), metadata);
  }

  async function complete(matrix) {
    if (!predictorReady) {
      if (typeof loadPyodide !== 'function') {
        await new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';
          script.onload = resolve;
          script.onerror = () => reject(
            new Error('Could not load the in-browser Python runtime.'));
          document.head.appendChild(script);
        });
      }
      pyodide = await loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
      });
      await pyodide.loadPackage(['numpy']);
      const response = await fetch('add-model/predictor.py?v=4');
      pyodide.FS.writeFile('predictor.py', await response.text());
      pyodide.runPython('import predictor');
      predictorReady = true;
    }
    pyodide.globals.set('MATRIX_JSON', JSON.stringify(matrix.observed));
    pyodide.globals.set('METRIC_JSON', JSON.stringify(
      matrix.benchmarks.map(benchmark => benchmark.metric.type)));
    pyodide.runPython(`
import json
matrix_values = json.loads(MATRIX_JSON)
metric_types = json.loads(METRIC_JSON)
completed_matrix = predictor.predict_matrix(matrix_values, metric_types)
del MATRIX_JSON, METRIC_JSON
`);
    const completedMatrix = pyodide.globals.get('completed_matrix');
    matrix.predictions = completedMatrix.toJs();
    completedMatrix.destroy();
    pyodide.runPython('del completed_matrix');
    return matrix;
  }

  return {load, readUpload, complete};
})();
