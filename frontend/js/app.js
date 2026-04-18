'use strict';
/* ═══════════════════════════════════════════════════════════════════
   INFERENCE — App Logic
   Wires command input, SSE stream, debate log output, agent state.
═══════════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────────────
const S = {
  sessionId:  null,
  isRunning:  false,
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

// ── Agent ID → list item map ──────────────────────────────────────────
const agentEls = {};
agentItems.forEach(el => { agentEls[el.dataset.agent] = el; });

// ── Keyboard shortcut ─────────────────────────────────────────────────
cmdInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') execute();
});
executeBtn.addEventListener('click', execute);

// ── File upload ───────────────────────────────────────────────────────
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

    // Auto-populate target col if we recognise common names
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

// ── Drag-and-drop anywhere ────────────────────────────────────────────
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

// ── Log helpers ───────────────────────────────────────────────────────
function log(text, cls = '') {
  // Remove initial "Ready." line on first real log
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

function setAgentState(agentId, state) {
  // state: 'idle' | 'running' | 'done' | 'error'
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
  S.isRunning    = running;
  executeBtn.disabled = running;
  cmdInput.disabled   = running;
  executeBtn.textContent = running ? 'RUNNING...' : 'EXECUTE';
}

// ── Score box ─────────────────────────────────────────────────────────
function showScore(report) {
  const score    = report.bias_score ?? 50;
  const severity = report.severity   || 'Unknown';

  scoreBox.classList.remove('hidden');
  scoreValue.textContent   = score;
  scoreSeverity.textContent = severity.toUpperCase();
  scoreSummary.textContent  = report.summary || '';

  // Colour coding
  scoreValue.className = '';
  if      (score <= 20) scoreValue.classList.add('low');
  else if (score <= 40) scoreValue.classList.add('low');
  else if (score <= 60) scoreValue.classList.add('moderate');
  else if (score <= 80) scoreValue.classList.add('high');
  else                  scoreValue.classList.add('severe');
}

function hideScore() {
  scoreBox.classList.add('hidden');
  scoreValue.textContent    = '--';
  scoreSeverity.textContent = '--';
  scoreSummary.textContent  = '';
  scoreValue.className      = '';
}

// ── SSE event handler ─────────────────────────────────────────────────
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

      // Mark agent as done (message received = turn complete)
      if (agentId && agentId !== 'final_judge') {
        setAgentState(agentId, 'done');
      } else if (agentId === 'final_judge') {
        setAgentState('final_judge', 'running');
      }

      // Write to debate log
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
      // handled in stream loop
      break;
  }
}

// ── Main execute ──────────────────────────────────────────────────────
async function execute() {
  const query = cmdInput.value.trim();
  if (!query || S.isRunning) return;

  // Reset state
  clearDebate();
  hideScore();
  resetAllAgents();
  setRunning(true);

  // Mark all analysts as running before stream starts
  ['data_statistician','fairness_auditor','domain_expert','bias_adversary','ethical_reviewer'].forEach(id => {
    setAgentState(id, 'running');
  });

  log(`> ${query}`, 'success');
  log(`> Model: ${modelSel.value}  Rounds: ${roundsSel.value}`, 'dim');
  log('> Initialising debate', 'dim');
  log('', 'dim'); // spacer

  const body = {
    query,
    session_id:           S.sessionId || '',
    use_case:             usecaseSel.value || 'general',
    protected_attributes: protectedIn.value.split(',').map(s => s.trim()).filter(Boolean),
    target_column:        targetColIn.value.trim(),
    model:                modelSel.value,
    max_rounds:           parseInt(roundsSel.value, 10),
  };

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

    // Parse SSE
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let lastEvent = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (line.startsWith('event:')) {
          lastEvent = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          try {
            const payload = JSON.parse(line.slice(5).trim());
            onSSE(lastEvent || 'message', payload);
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
  }
}

// ── Init ──────────────────────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error();
  } catch {
    log('> WARNING: Backend not reachable.', 'error');
  }
})();
