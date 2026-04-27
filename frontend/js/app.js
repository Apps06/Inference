'use strict';
/* ═══════════════════════════════════════════════════════════════════
   INFERENCE — App Logic v2
   New in v2:
   - Tab switching: DEBATE ↔ HISTORY
   - Trajectory history panel with stats
   - RL reward breakdown display (with animated bars)
   - Animated bias score counter
   - Full model dropdown (grok-4.1-fast-non-reasoning added)
═══════════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────────────
const S = {
  sessionId:       null,
  isRunning:       false,
  activeTab:       'debate',
  selectedDebateId: null,
};

// ── DOM ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const cmdInput      = $('cmd-input');
const executeBtn    = $('execute-btn');
const debateOutput  = $('debate-output');
const modelSel      = $('model-select');
const roundsSel     = $('rounds-select');
const usecaseSel    = $('usecase-select');
const targetColIn   = $('target-col-input');
const protectedIn   = $('protected-input');
const fileInput     = $('file-input');
const fileLabel     = $('file-label');
const fileLabelText = $('file-label-text');
const agentItems    = document.querySelectorAll('.agent-item');
const scoreBox      = $('score-box');
const scoreValue    = $('score-value');
const scoreSeverity = $('score-severity');
const scoreSummary  = $('score-summary');
const rewardBox     = $('reward-box');
const rewardList    = $('reward-list');
const rewardTotal   = $('reward-total-line');
const historyList   = $('history-list');
const historyStats  = $('history-stats');

// ── Agent ID → list item map ──────────────────────────────────────────
const agentEls = {};
agentItems.forEach(el => { agentEls[el.dataset.agent] = el; });

// ── Reward component display names ───────────────────────────────────
const REWARD_LABELS = {
  fairness_coverage:        'Fairness Coverage',
  report_quality:           'Report Quality',
  debate_depth:             'Debate Depth',
  legal_completeness:       'Legal Completeness',
  mitigation_actionability: 'Mitigation Quality',
};

// ═══════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    if (tab === S.activeTab) return;
    S.activeTab = tab;

    document.querySelectorAll('.tab-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.tab === tab)
    );
    document.querySelectorAll('.tab-panel').forEach(p =>
      p.classList.toggle('active', p.dataset.tab === tab)
    );

    if (tab === 'history') loadHistory();
  });
});

// ═══════════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUT + EXECUTE BUTTON
// ═══════════════════════════════════════════════════════════════════

cmdInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') execute();
});
executeBtn.addEventListener('click', execute);

// ═══════════════════════════════════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════════════════════════════════

$('file-label').addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  fileLabelText.textContent = '[ UPLOADING... ]';

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await res.json();
    S.sessionId = data.session_id;

    fileLabelText.textContent = `[ ${file.name.toUpperCase()} — ${data.rows} ROWS ]`;
    fileLabel.classList.add('has-file');

    // Auto-populate target col
    if (!targetColIn.value) {
      const guess = data.columns.find(c =>
        ['hired','approved','outcome','label','target','result','decision'].includes(c.toLowerCase())
      );
      if (guess) targetColIn.value = guess;
    }

    log(`> Dataset loaded: ${data.rows} rows, ${data.columns.length} columns.`, 'success');
    log(`> Columns: ${data.columns.join(', ')}`, 'dim');
  } catch (err) {
    fileLabelText.textContent = '[ UPLOAD FAILED ]';
    fileLabel.classList.remove('has-file');
    log(`> ERROR: ${err.message}`, 'error');
  }
});

// Drag-and-drop anywhere
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file?.name.toLowerCase().endsWith('.csv')) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

// ═══════════════════════════════════════════════════════════════════
// LOG HELPERS
// ═══════════════════════════════════════════════════════════════════

function log(text, cls = '') {
  const initLine = $('init-line');
  if (initLine) initLine.remove();

  const span = document.createElement('span');
  span.className = 'log-line' + (cls ? ' ' + cls : '');
  span.textContent = text;
  debateOutput.appendChild(span);
  debateOutput.scrollTop = debateOutput.scrollHeight;
  return span;
}

function clearDebate() {
  debateOutput.innerHTML = '';
}

// ═══════════════════════════════════════════════════════════════════
// AGENT STATE
// ═══════════════════════════════════════════════════════════════════

function setAgentState(agentId, state) {
  const el = agentEls[agentId];
  if (!el) return;
  el.className = 'agent-item' + (state !== 'idle' ? ' ' + state : '');
  const statusEl = el.querySelector('.agent-status');
  statusEl.className = 'agent-status ' + state;
  statusEl.textContent = state.toUpperCase();
}

function resetAllAgents() {
  Object.keys(agentEls).forEach(id => setAgentState(id, 'idle'));
}

function setRunning(running) {
  S.isRunning         = running;
  executeBtn.disabled = running;
  cmdInput.disabled   = running;
  executeBtn.textContent = running ? 'RUNNING...' : 'EXECUTE';
}

// ═══════════════════════════════════════════════════════════════════
// SCORE BOX
// ═══════════════════════════════════════════════════════════════════

function _scoreClass(score) {
  if (score <= 20) return 'low';
  if (score <= 40) return 'low';
  if (score <= 60) return 'moderate';
  if (score <= 80) return 'high';
  return 'severe';
}

function _animateCount(el, targetVal, durationMs = 700) {
  const start    = performance.now();
  const startVal = 0;
  function frame(now) {
    const t = Math.min((now - start) / durationMs, 1);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    el.textContent = Math.round(startVal + (targetVal - startVal) * eased);
    if (t < 1) requestAnimationFrame(frame);
    else el.textContent = targetVal;
  }
  requestAnimationFrame(frame);
}

function showScore(report) {
  const score    = report.bias_score ?? 50;
  const severity = report.severity   || 'Unknown';

  scoreBox.classList.remove('hidden');
  scoreValue.className = _scoreClass(score);
  _animateCount(scoreValue, score);
  scoreSeverity.textContent = severity.toUpperCase();
  scoreSummary.textContent  = report.summary || '';
}

function showReward(reward) {
  if (!reward || typeof reward.total !== 'number') return;

  rewardBox.classList.remove('hidden');
  rewardList.innerHTML = '';

  const components = Object.entries(REWARD_LABELS);
  for (const [key, label] of components) {
    const val = reward[key];
    if (typeof val !== 'number') continue;
    const pct = Math.round(val * 100);

    const li = document.createElement('li');
    li.className = 'reward-item';
    li.innerHTML = `
      <span class="reward-key">${label}</span>
      <span class="reward-bar-wrap"><span class="reward-bar" style="width:0%"></span></span>
      <span class="reward-val">${pct}%</span>
    `;
    rewardList.appendChild(li);

    // Animate bar after paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        li.querySelector('.reward-bar').style.width = `${pct}%`;
      });
    });
  }

  const totalPct = Math.round(reward.total * 100);
  rewardTotal.innerHTML = `
    <span>TOTAL RL REWARD</span>
    <span style="color:var(--white);font-family:var(--font-title);font-size:13px;font-weight:900">${totalPct}%</span>
  `;
}

function hideScore() {
  scoreBox.classList.add('hidden');
  rewardBox.classList.add('hidden');
  scoreValue.textContent    = '--';
  scoreSeverity.textContent = '--';
  scoreSummary.textContent  = '';
  scoreValue.className      = '';
  rewardList.innerHTML      = '';
  rewardTotal.innerHTML     = '';
}

// ═══════════════════════════════════════════════════════════════════
// SSE EVENT HANDLER
// ═══════════════════════════════════════════════════════════════════

function onSSE(eventType, data) {
  switch (eventType) {
    case 'round_start': {
      const n = (data.round ?? 0) + 1;
      log(`\n── ROUND ${n} ${'─'.repeat(40)}`, 'round-header');
      break;
    }

    case 'agent_message': {
      const agentId = data.agent;
      const role    = data.content === 'Synthesizing all arguments...'
        ? data.role
        : data.role || agentId;

      if (agentId && agentId !== 'final_judge') {
        setAgentState(agentId, 'done');
      } else if (agentId === 'final_judge') {
        setAgentState('final_judge', 'running');
      }

      log(`${role}:`, 'agent-header');
      log(data.content || '', 'agent-body');
      break;
    }

    case 'final_report': {
      setAgentState('final_judge', 'done');
      log('\n── FINAL REPORT ──────────────────────────────', 'round-header');

      if (data.flagged_issues?.length) {
        log(`FLAGGED ISSUES (${data.flagged_issues.length}):`, 'agent-header');
        data.flagged_issues.forEach(issue => {
          log(`  [${(issue.severity || '').toUpperCase()}] ${issue.issue}`, 'agent-body');
          if (issue.evidence) log(`    Evidence: ${issue.evidence}`, 'dim');
        });
      }

      if (data.mitigation_steps?.length) {
        log('MITIGATION STEPS:', 'agent-header');
        data.mitigation_steps.forEach((m, i) => {
          log(`  ${i + 1}. [${(m.priority || '').toUpperCase()}] ${m.step}`, 'agent-body');
        });
      }

      if (data.debate_id) {
        log(`\nTrajectory ID: ${data.debate_id}`, 'dim');
      }

      showScore(data);
      // Reward breakdown will be fetched separately after stream ends
      break;
    }

    case 'error':
      log(`> ERROR: ${data.message || 'Unknown error'}`, 'error');
      Object.keys(agentEls).forEach(id => {
        const el = agentEls[id];
        if (el.classList.contains('running')) setAgentState(id, 'error');
      });
      break;

    case 'done':
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN EXECUTE
// ═══════════════════════════════════════════════════════════════════

async function execute() {
  const query = cmdInput.value.trim();
  if (!query || S.isRunning) return;

  clearDebate();
  hideScore();
  resetAllAgents();
  setRunning(true);

  ['data_statistician','fairness_auditor','domain_expert','bias_adversary','ethical_reviewer'].forEach(id => {
    setAgentState(id, 'running');
  });

  log(`> ${query}`, 'success');
  log(`> Model: ${modelSel.value}  Rounds: ${roundsSel.value}`, 'dim');
  log('> Initialising debate', 'dim');
  log('', 'dim');

  const body = {
    query,
    session_id:           S.sessionId || '',
    use_case:             usecaseSel.value || 'general',
    protected_attributes: protectedIn.value.split(',').map(s => s.trim()).filter(Boolean),
    target_column:        targetColIn.value.trim(),
    model:                modelSel.value,
    max_rounds:           parseInt(roundsSel.value, 10),
  };

  let lastDebateId = null;

  try {
    const res = await fetch('/api/chat/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';
    let lastEvent = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('event:')) {
          lastEvent = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          try {
            const payload = JSON.parse(line.slice(5).trim());
            onSSE(lastEvent || 'message', payload);
            if (payload.debate_id) lastDebateId = payload.debate_id;
          } catch { /* malformed line */ }
          if (lastEvent === 'done') break;
          lastEvent = null;
        }
      }
    }
  } catch (err) {
    log(`> FATAL: ${err.message}`, 'error');
  } finally {
    setRunning(false);
    log('\n> Session complete.', 'dim');

    // Fetch reward breakdown for the just-completed debate
    if (lastDebateId) {
      try {
        const rec = await fetch(`/api/trajectories/${lastDebateId}`).then(r => r.json());
        if (rec?.reward) showReward(rec.reward);
      } catch (_) { /* reward display is non-critical */ }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// HISTORY PANEL
// ═══════════════════════════════════════════════════════════════════

async function loadHistory() {
  historyList.innerHTML  = '<span class="log-line dim">Loading...</span>';
  historyStats.innerHTML = '<span class="log-line dim">Loading stats...</span>';

  try {
    const [list, stats] = await Promise.all([
      fetch('/api/trajectories').then(r => r.json()),
      fetch('/api/trajectories/stats').then(r => r.json()),
    ]);

    renderHistoryList(list);
    renderHistoryStats(stats);
  } catch (err) {
    historyList.innerHTML  = `<span class="log-line error">> ERROR: ${err.message}</span>`;
    historyStats.innerHTML = '';
  }
}

function _scoreCls(score) {
  if (score == null) return 'unknown';
  if (score <= 40)   return 'low';
  if (score <= 60)   return 'moderate';
  if (score <= 80)   return 'high';
  return 'severe';
}

function renderHistoryList(list) {
  if (!list || list.length === 0) {
    historyList.innerHTML = '<span class="log-line dim">No debates logged yet. Run a debate first.</span>';
    return;
  }

  historyList.innerHTML = '';
  list.forEach(item => {
    const ts  = item.timestamp
      ? new Date(item.timestamp).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })
      : '—';
    const scoreNum  = item.bias_score ?? '—';
    const scoreCls  = _scoreCls(item.bias_score);
    const rwdPct    = typeof item.total_reward === 'number'
      ? `RL: ${Math.round(item.total_reward * 100)}%`
      : '';
    const attrs     = (item.protected_attributes || []).join(', ') || '—';

    const div = document.createElement('div');
    div.className = 'history-item log-line';
    div.innerHTML = `
      <div class="h-meta">
        <span class="h-usecase">${(item.use_case || 'general').toUpperCase()}</span>
        <span class="h-score ${scoreCls}">${scoreNum}</span>
        <span class="h-severity">${(item.severity || '—').toUpperCase()}</span>
        <span class="h-reward">${rwdPct}</span>
      </div>
      <div class="h-meta" style="margin-top:3px">
        <span class="h-timestamp">${ts}</span>
        <span class="h-timestamp">attrs: ${attrs}</span>
      </div>
    `;

    div.addEventListener('click', () => {
      document.querySelectorAll('.history-item').forEach(el => el.classList.remove('selected'));
      div.classList.add('selected');
      S.selectedDebateId = item.debate_id;
      loadDebateDetail(item.debate_id);
    });

    historyList.appendChild(div);
  });
}

function renderHistoryStats(stats) {
  if (!stats || stats.total_debates === 0) {
    historyStats.innerHTML = '<span class="log-line dim">No stats available yet.</span>';
    return;
  }

  const rows = [
    ['TOTAL DEBATES',   stats.total_debates],
    ['AVG BIAS SCORE',  stats.avg_bias_score ?? '—'],
    ['AVG RL REWARD',   typeof stats.avg_total_reward === 'number'
      ? `${Math.round(stats.avg_total_reward * 100)}%` : '—'],
  ];

  let html = rows.map(([k, v]) => `
    <div class="stats-row">
      <span class="stats-key">${k}</span>
      <span class="stats-value">${v}</span>
    </div>
  `).join('');

  // Use-case breakdown
  if (stats.use_case_breakdown && Object.keys(stats.use_case_breakdown).length > 0) {
    html += '<div class="stats-section-header">USE CASE BREAKDOWN</div>';
    for (const [uc, n] of Object.entries(stats.use_case_breakdown)) {
      html += `
        <div class="stats-row">
          <span class="stats-key">${uc.toUpperCase()}</span>
          <span class="stats-value">${n}</span>
        </div>`;
    }
  }

  // Model breakdown
  if (stats.model_breakdown && Object.keys(stats.model_breakdown).length > 0) {
    html += '<div class="stats-section-header">MODEL BREAKDOWN</div>';
    for (const [model, n] of Object.entries(stats.model_breakdown)) {
      html += `
        <div class="stats-row">
          <span class="stats-key">${model.toUpperCase()}</span>
          <span class="stats-value">${n}</span>
        </div>`;
    }
  }

  // Per-agent reward means
  const par = stats.avg_per_agent_rewards;
  if (par && Object.keys(par).length > 0) {
    html += '<div class="stats-section-header">AVG AGENT REWARDS</div>';
    for (const [agentId, rwds] of Object.entries(par)) {
      const label = agentId.replace(/_/g, ' ').toUpperCase();
      const total = typeof rwds.total === 'number' ? `${Math.round(rwds.total * 100)}%` : '—';
      html += `
        <div class="stats-row">
          <span class="stats-key">${label}</span>
          <span class="stats-value">${total}</span>
        </div>`;
    }
  }

  historyStats.innerHTML = html;
}

async function loadDebateDetail(debateId) {
  historyStats.innerHTML = '<span class="log-line dim">Loading debate...</span>';
  try {
    const rec = await fetch(`/api/trajectories/${debateId}`).then(r => r.json());
    renderDebateDetail(rec);
  } catch (err) {
    historyStats.innerHTML = `<span class="log-line error">> ERROR: ${err.message}</span>`;
  }
}

function renderDebateDetail(rec) {
  const rpt     = rec.final_report   || {};
  const reward  = rec.reward         || {};
  const agents  = rec.per_agent_rewards || {};

  const ts = rec.timestamp
    ? new Date(rec.timestamp).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })
    : '—';

  let html = `
    <div class="stats-row"><span class="stats-key">DEBATE ID</span>
      <span class="stats-value" style="font-size:9px;word-break:break-all">${rec.debate_id?.slice(0,12) || '—'}…</span></div>
    <div class="stats-row"><span class="stats-key">TIME</span><span class="stats-value">${ts}</span></div>
    <div class="stats-row"><span class="stats-key">MODEL</span><span class="stats-value">${(rec.model_backend || '—').toUpperCase()}</span></div>
    <div class="stats-row"><span class="stats-key">USE CASE</span><span class="stats-value">${(rec.use_case || '—').toUpperCase()}</span></div>
    <div class="stats-row"><span class="stats-key">BIAS SCORE</span>
      <span class="stats-value h-score ${_scoreCls(rpt.bias_score)}">${rpt.bias_score ?? '—'}</span></div>
    <div class="stats-row"><span class="stats-key">SEVERITY</span><span class="stats-value">${(rpt.severity || '—').toUpperCase()}</span></div>
    <div class="stats-row"><span class="stats-key">LEGAL RISK</span><span class="stats-value">${(rpt.legal_risk || '—').toUpperCase()}</span></div>
    <div class="stats-row"><span class="stats-key">CONFIDENCE</span><span class="stats-value">${(rpt.confidence || '—').toUpperCase()}</span></div>
  `;

  // Global reward
  html += '<div class="stats-section-header">GLOBAL RL REWARD</div>';
  const globalKeys = ['fairness_coverage','report_quality','debate_depth','legal_completeness','mitigation_actionability','total'];
  for (const k of globalKeys) {
    if (typeof reward[k] !== 'number') continue;
    const label = (REWARD_LABELS[k] || k).toUpperCase();
    const pct   = Math.round(reward[k] * 100);
    const bold  = k === 'total' ? 'font-weight:900;color:var(--white)' : '';
    html += `<div class="stats-row"><span class="stats-key">${label}</span>
      <span class="stats-value" style="${bold}">${pct}%</span></div>`;
  }

  // Per-agent reward totals
  if (Object.keys(agents).length > 0) {
    html += '<div class="stats-section-header">PER-AGENT REWARDS</div>';
    for (const [agentId, rwds] of Object.entries(agents)) {
      const label = agentId.replace(/_/g, ' ').toUpperCase();
      const total = typeof rwds.total === 'number' ? `${Math.round(rwds.total * 100)}%` : '—';
      html += `<div class="stats-row"><span class="stats-key">${label}</span>
        <span class="stats-value">${total}</span></div>`;
    }
  }

  historyStats.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════
(async () => {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error();
    // Pre-warm: try to load stats silently for first history visit
    fetch('/api/trajectories/stats').catch(() => {});
  } catch {
    log('> WARNING: Backend not reachable.', 'error');
  }
})();
