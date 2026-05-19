(async () => {
  const res = await fetch('data.json?v=7');
  const data = await res.json();

  const modelInput = document.getElementById('model');
  const benchInput = document.getElementById('bench');
  const modelList = document.getElementById('model-list');
  const benchList = document.getElementById('bench-list');
  const bigval = document.getElementById('bigval');
  const bigsub = document.getElementById('bigsub');
  const bigEyebrow = document.getElementById('big-eyebrow');
  const big = document.getElementById('big');
  const toprows = document.getElementById('toprows');

  const ms = data.models.map((m,i)=>({...m,idx:i})).sort((a,b)=>a.name.localeCompare(b.name));
  const bs = data.benchmarks.map((b,j)=>({...b,idx:j})).sort((a,b)=>a.name.localeCompare(b.name));
  const modelLabel = m => `${m.name} — ${m.provider}`;
  const benchLabel = b => b.name;
  const modelByLabel = new Map();
  const benchByLabel = new Map();
  for (const m of ms) {
    const o = document.createElement('option');
    o.value = modelLabel(m);
    modelList.appendChild(o);
    modelByLabel.set(o.value, m.idx);
  }
  for (const b of bs) {
    const o = document.createElement('option');
    o.value = benchLabel(b);
    benchList.appendChild(o);
    benchByLabel.set(o.value, b.idx);
  }

  let currentI = 0, currentJ = 0;

  function update() {
    const i = currentI, j = currentJ;
    const m = data.models[i], b = data.benchmarks[j];
    const pred = data.predictions[i][j], obs = data.observed[i][j];
    const interval = data.prediction_intervals?.[i]?.[j];
    const trustProbability = data.trust_probabilities?.[i]?.[j];

    bigval.classList.remove('observed');
    big.querySelector('.bigtag')?.remove();

    if (obs !== null) {
      bigEyebrow.textContent = 'Reported score';
      bigval.textContent = obs.toFixed(1);
      bigval.classList.add('observed');
      bigsub.innerHTML = `<strong>${m.name}</strong> on ${b.name}`;
      const t = document.createElement('div'); t.className = 'bigtag observed'; t.textContent = 'Reported';
      bigsub.after(t);
    } else if (pred !== null) {
      bigEyebrow.textContent = 'Predicted score + trust';
      bigval.textContent = pred.toFixed(1);
      bigsub.innerHTML = `<strong>${m.name}</strong> on ${b.name}<br>No public score reported${trustLine(trustProbability)}${interval ? `<br><span class="interval-text">90% predicted range: ${formatInterval(interval)}</span>` : ''}`;
      const t = document.createElement('div'); t.className = 'bigtag predicted'; t.textContent = 'Predicted';
      bigsub.after(t);
    } else {
      bigEyebrow.textContent = 'Score';
      bigval.textContent = '—';
      bigsub.textContent = 'No prediction available.';
    }

    renderTop(j, i);
  }

  let lbMode = 'full';
  let expandedI = null;

  function formatInterval(interval) {
    if (!interval || interval.length !== 2 || interval[0] === null || interval[1] === null) return '—';
    return `${interval[0].toFixed(1)}–${interval[1].toFixed(1)}`;
  }

  function formatTrustProbability(probability) {
    return Number.isFinite(probability) ? `${Math.round(100 * probability)}%` : '—';
  }

  function trustLine(probability) {
    return Number.isFinite(probability)
      ? `<br><span class="interval-text">Trust probability: ${formatTrustProbability(probability)}</span>`
      : '';
  }

  function intervalLine(i, j, isObserved) {
    const interval = data.prediction_intervals?.[i]?.[j];
    if (!interval) return '';
    const prefix = isObserved ? 'BenchPress 90% predicted range' : '90% predicted range';
    return `<div class="meta-line interval-text">${prefix}: ${formatInterval(interval)}</div>`;
  }

  function settingLines(setting) {
    if (!setting) return '';
    const order = ['mode','effort','tools','sampling','judge','harness','prompt_style','temperature','context','notes'];
    const items = [];
    for (const k of order) {
      if (setting[k] === undefined || setting[k] === null || setting[k] === '') continue;
      items.push(`<div class="kv"><span class="k">${k}</span><span class="v">${String(setting[k])}</span></div>`);
    }
    return items.join('');
  }

  function rowDetails(r) {
    const m = data.models[r.i];
    const parts = [];
    const reason = m.reasoning ? 'reasoning model' : 'non-reasoning model';
    const rel = m.release_date ? ` · released ${m.release_date}` : '';
    const ow = m.open_weights === true ? ' · open weights' : (m.open_weights === false ? ' · closed weights' : '');
    parts.push(`<div class="meta-line">${reason}${rel}${ow}</div>`);

    if (r.obs !== null) {
      const src = data.sources?.[r.i]?.[currentJ];
      if (src) {
        const linkLine = src.url
          ? `<div class="meta-line"><a href="${src.url}" target="_blank" rel="noopener">↗ source link</a>${src.matches_canonical === false ? ' · non-canonical setting' : ''}${src.audit_status ? ' · audit: ' + src.audit_status : ''}</div>`
          : '';
        parts.push(linkLine);
        if (src.reported_setting) {
          parts.push(`<div class="kv-block"><div class="kv-title">Reported setting</div>${settingLines(src.reported_setting)}</div>`);
        }
        if (src.notes) parts.push(`<div class="notes">${src.notes}</div>`);
      } else {
        parts.push(`<div class="meta-line dim">Source metadata unavailable.</div>`);
      }
    } else {
      parts.push(`<div class="meta-line dim">BenchPress prediction (Logit Bias ALS, rank=2, λ=0.1). No public score reported.</div>`);
      const trustProbability = data.trust_probabilities?.[r.i]?.[currentJ];
      if (Number.isFinite(trustProbability)) {
        parts.push(`<div class="meta-line interval-text">Trust probability: ${formatTrustProbability(trustProbability)}</div>`);
      }
      parts.push(intervalLine(r.i, currentJ, false));
      if (m.canonical_setting) {
        parts.push(`<div class="kv-block"><div class="kv-title">Model canonical setting (used as covariate)</div>${settingLines(m.canonical_setting)}</div>`);
      }
    }
    return parts.join('');
  }

  function renderTop(j, currentI) {
    const b = data.benchmarks[j];
    document.getElementById('lb-title').textContent = `On ${b.name}.`;
    const all = data.models.map((m,i)=>({
      i, name: m.name, provider: m.provider,
      pred: data.predictions[i][j], obs: data.observed[i][j],
    }));
    let rows = all.filter(r => {
      if (lbMode === 'obs') return r.obs !== null;
      const v = r.obs ?? r.pred;
      return v !== null && v >= 0 && v <= 100;
    });
    rows.sort((a,b)=>(b.obs ?? b.pred) - (a.obs ?? a.pred));
    const rankIdx = rows.findIndex(r => r.i === currentI);
    const myRank = rankIdx >= 0 ? rankIdx + 1 : null;
    const myRankEl = document.getElementById('lb-myrank');
    const curName = data.models[currentI].name;
    if (myRank) {
      myRankEl.innerHTML = `<strong>${curName}</strong> ranks <strong>#${myRank}</strong> of ${rows.length}.`;
    } else {
      myRankEl.innerHTML = `<strong>${curName}</strong> has no ${lbMode === 'obs' ? 'reported' : ''} score on this benchmark.`;
    }
    toprows.innerHTML = rows.map((r,k)=>{
      const isObs = r.obs !== null, val = isObs ? r.obs : r.pred;
      const cls = isObs ? 'observed' : 'predicted';
      const hl = r.i === currentI ? ' rowitem-current' : '';
      const exp = r.i === expandedI ? ' rowitem-expanded' : '';
      const src = isObs ? data.sources?.[r.i]?.[currentJ] : null;
      const interval = data.prediction_intervals?.[r.i]?.[j];
      const intervalHtml = (!isObs && interval)
        ? `<div class="val-interval">${formatInterval(interval)}</div>`
        : '';
      const trustProbability = data.trust_probabilities?.[r.i]?.[j];
      const trustHtml = (!isObs && Number.isFinite(trustProbability))
        ? `<div class="val-interval">trust ${formatTrustProbability(trustProbability)}</div>`
        : '';
      const srcLink = (src && src.url)
        ? `<a class="srclink" href="${src.url}" target="_blank" rel="noopener" title="Open ${src.type || 'source'}" onclick="event.stopPropagation()">↗</a>`
        : `<span class="srclink dim" title="${isObs ? 'No source URL' : 'BenchPress prediction'}">${isObs ? '·' : '~'}</span>`;
      const chev = `<div class="chev">▾</div>`;
      return `<div class="rowitem${hl}${exp}" data-i="${r.i}">
        <div class="rowmain">
          <div class="rank">#${k+1}</div>
          <div class="name">${r.name}</div>
          <div class="tag ${cls}">${isObs ? 'reported' : 'predicted'}</div>
          <div class="valwrap"><div class="val ${cls}">${val.toFixed(1)}</div>${trustHtml}${intervalHtml}</div>
          ${chev}
        </div>
        <div class="rowdetails">${r.i === expandedI ? rowDetails(r) : ''}</div>
      </div>`;
    }).join('');
  }

  toprows.addEventListener('click', (e) => {
    const row = e.target.closest('.rowitem');
    if (!row) return;
    const i = +row.dataset.i;
    expandedI = (expandedI === i) ? null : i;
    renderTop(currentJ, currentI);
  });

  document.querySelectorAll('.lb-mode').forEach(el => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      lbMode = el.dataset.mode;
      document.querySelectorAll('.lb-mode').forEach(x => x.classList.toggle('active', x === el));
      update();
    });
  });

  function makeCombo(input, byLabel, getLabel, getCurrent, setCurrent) {
    function commit() {
      const v = input.value.trim();
      if (byLabel.has(v)) { setCurrent(byLabel.get(v)); update(); }
    }
    input.addEventListener('input', commit);
    input.addEventListener('change', commit);
    input.addEventListener('focus', () => {
      input.dataset.prev = input.value;
      input.value = '';
    });
    input.addEventListener('blur', () => {
      const v = input.value.trim();
      if (!byLabel.has(v)) {
        input.value = input.dataset.prev || getLabel(getCurrent());
      }
    });
  }
  makeCombo(modelInput, modelByLabel, idx => modelLabel(data.models[idx]),
    () => currentI, idx => currentI = idx);
  makeCombo(benchInput, benchByLabel, idx => benchLabel(data.benchmarks[idx]),
    () => currentJ, idx => currentJ = idx);

  const dm = ms.findIndex(m => m.name.toLowerCase().includes('gpt-5'));
  const db = bs.findIndex(b => b.name.toLowerCase().includes('gpqa'));
  const defM = dm>=0 ? ms[dm] : ms[0];
  const defB = db>=0 ? bs[db] : bs[0];
  currentI = defM.idx; currentJ = defB.idx;
  modelInput.value = modelLabel(defM);
  benchInput.value = benchLabel(defB);
  update();
})();
