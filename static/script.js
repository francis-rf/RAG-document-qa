// ── State ──────────────────────────────────────────────────────
const state = { indexed: false, busy: false };

// ── DOM refs ───────────────────────────────────────────────────
const sidebar       = document.getElementById('sidebar');
const overlay       = document.getElementById('overlay');
const menuBtn       = document.getElementById('menuBtn');
const uploadArea    = document.getElementById('uploadArea');
const fileInput     = document.getElementById('fileInput');
const uploadLink    = document.getElementById('uploadLink');
const refreshBtn    = document.getElementById('refreshBtn');
const fileList      = document.getElementById('fileList');
const indexBtn      = document.getElementById('indexBtn');
const indexStatus   = document.getElementById('indexStatus');
const indexBadge    = document.getElementById('indexBadge');
const messages      = document.getElementById('messages');
const welcome       = document.getElementById('welcome');
const chatWrap      = document.getElementById('chatWrap');
const questionInput = document.getElementById('questionInput');
const sendBtn       = document.getElementById('sendBtn');

// ── Sidebar ────────────────────────────────────────────────────
menuBtn.addEventListener('click', toggleSidebar);
overlay.addEventListener('click', closeSidebar);

function toggleSidebar() {
  if (window.innerWidth <= 768) {
    sidebar.classList.toggle('open');
    overlay.classList.toggle('active');
  } else {
    sidebar.classList.toggle('collapsed');
  }
}

function closeSidebar() {
  sidebar.classList.remove('open');
  overlay.classList.remove('active');
}

// Close sidebar on mobile after action
function closeSidebarOnMobile() {
  if (window.innerWidth <= 768) {
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
  }
}

// ── File upload ────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());
uploadLink.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const files = [...e.dataTransfer.files].filter(f => f.type === 'application/pdf');
  if (files.length) uploadFiles(files);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFiles([...fileInput.files]);
  fileInput.value = '';
});

async function uploadFiles(files) {
  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        console.error('Upload failed:', err.detail);
      }
    } catch (err) {
      console.error('Upload error:', err);
    }
  }
  loadFileList();
}

// ── File list ──────────────────────────────────────────────────
async function loadFileList() {
  try {
    const res = await fetch('/api/files');
    const data = await res.json();
    renderFileList(data.files);
  } catch {
    renderFileList([]);
  }
}

function renderFileList(files) {
  fileList.innerHTML = '';
  if (!files || files.length === 0) {
    fileList.innerHTML = '<li class="file-empty">No files yet</li>';
    return;
  }
  files.forEach(name => {
    const li = document.createElement('li');
    li.className = 'file-item';
    li.textContent = name;
    li.title = name;
    fileList.appendChild(li);
  });
}

refreshBtn.addEventListener('click', loadFileList);

// ── Index documents ────────────────────────────────────────────
indexBtn.addEventListener('click', async () => {
  if (state.busy) return;
  state.busy = true;
  indexBtn.disabled = true;
  showStatus('loading', 'Indexing documents…');

  try {
    const res = await fetch('/api/load', { method: 'POST' });
    const data = await res.json();

    if (data.status === 'success') {
      showStatus('success', data.message);
      state.indexed = true;
      indexBadge.classList.add('visible');
      updateSendBtn();
    } else {
      showStatus('error', data.message);
    }
  } catch {
    showStatus('error', 'Failed to index documents. Is the server running?');
  } finally {
    state.busy = false;
    indexBtn.disabled = false;
  }
});

function showStatus(type, msg) {
  indexStatus.className = `status-pill ${type} visible`;
  indexStatus.textContent = msg;
}

// ── Chat input ─────────────────────────────────────────────────
questionInput.addEventListener('input', () => {
  autoResize(questionInput);
  updateSendBtn();
});

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

function updateSendBtn() {
  sendBtn.disabled = !questionInput.value.trim();
}

// ── Send message ───────────────────────────────────────────────
async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || state.busy) return;

  welcome.classList.add('hidden');
  questionInput.value = '';
  autoResize(questionInput);
  sendBtn.disabled = true;
  state.busy = true;

  appendUserMessage(question);
  const typingEl = appendTyping();
  scrollBottom();

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    typingEl.remove();

    if (!res.ok) {
      const err = await res.json();
      appendAIMessage(err.detail || 'An error occurred.', []);
    } else {
      const data = await res.json();
      appendAIMessage(data.answer, data.retrieved_docs);
    }
  } catch {
    typingEl.remove();
    appendAIMessage('Failed to get a response. Please try again.', []);
  } finally {
    state.busy = false;
    updateSendBtn();
    scrollBottom();
  }
}

// ── Message rendering ──────────────────────────────────────────
function appendUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `
    <div class="msg-label">You</div>
    <div class="msg-bubble">${escapeHtml(text)}</div>
  `;
  messages.appendChild(el);
}

function appendAIMessage(text, sources) {
  const el = document.createElement('div');
  el.className = 'msg ai';

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const cards = sources.map(s => `
      <div class="source-card">
        <div class="source-meta">
          <span class="source-name">${escapeHtml(s.source)}</span>
          <span class="source-page">p. ${s.page}</span>
        </div>
        <div class="source-preview">${escapeHtml(s.content)}…</div>
      </div>
    `).join('');

    sourcesHtml = `
      <div class="sources">
        <button class="sources-toggle">
          <svg class="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="9 18 15 12 9 6"/>
          </svg>
          ${sources.length} source${sources.length > 1 ? 's' : ''}
        </button>
        <div class="sources-list">${cards}</div>
      </div>
    `;
  }

  el.innerHTML = `
    <div class="msg-label">Assistant</div>
    <div class="msg-bubble">${formatText(text)}</div>
    ${sourcesHtml}
  `;

  el.querySelector('.sources-toggle')?.addEventListener('click', function () {
    this.classList.toggle('open');
    this.nextElementSibling.classList.toggle('open');
  });

  messages.appendChild(el);
}

function appendTyping() {
  const el = document.createElement('div');
  el.className = 'msg ai';
  el.innerHTML = `
    <div class="msg-label">Assistant</div>
    <div class="msg-bubble typing-dots">
      <span></span><span></span><span></span>
    </div>
  `;
  messages.appendChild(el);
  return el;
}

// ── Helpers ────────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function scrollBottom() {
  chatWrap.scrollTop = chatWrap.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// ── Init ───────────────────────────────────────────────────────
loadFileList();
