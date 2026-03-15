const state = {
  sessionId: null,
  algorithm: "dqn",
  speechEnabled: true,
  recognition: null,
  listening: false,
};

const elements = {
  algorithmSelect: document.getElementById("algorithm-select"),
  resetButton: document.getElementById("reset-button"),
  chatThread: document.getElementById("chat-thread"),
  messageInput: document.getElementById("message-input"),
  sendButton: document.getElementById("send-button"),
  listenButton: document.getElementById("listen-button"),
  speakToggle: document.getElementById("speak-toggle"),
  voiceStatus: document.getElementById("voice-status"),
  summaryCards: document.getElementById("summary-cards"),
  slotProgress: document.getElementById("slot-progress"),
  concernList: document.getElementById("concern-list"),
  focusList: document.getElementById("focus-list"),
  summaryMarkdown: document.getElementById("summary-markdown"),
};

async function requestJson(url, payload = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function speak(text) {
  if (!state.speechEnabled || !("speechSynthesis" in window) || !text) {
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "zh-TW";
  utterance.rate = 1;
  window.speechSynthesis.speak(utterance);
}

function setVoiceStatus(text) {
  elements.voiceStatus.textContent = text;
}

function renderEmptyState() {
  elements.chatThread.innerHTML = `
    <div class="empty-state">
      <strong>正在建立照護對話 session</strong>
      <p>稍等一下，系統會先選好第一段陪伴話術。</p>
    </div>
  `;
}

function renderChat(payload) {
  const turns = payload.turns || [];
  const latestPrompt = payload.latest_assistant_message;
  elements.chatThread.innerHTML = "";

  if (turns.length === 0 && latestPrompt) {
    elements.chatThread.appendChild(createBubble("assistant", latestPrompt, ["opening", payload.current_target_slot]));
  } else {
    turns.forEach((turn) => {
      elements.chatThread.appendChild(
        createBubble("assistant", turn.assistant_message, [`script ${turn.script_id}`, turn.target_slot])
      );
      const tags = [`deviation ${turn.deviation_level}`];
      if (turn.transition_used) {
        tags.push("transition");
      }
      elements.chatThread.appendChild(createBubble("user", turn.elder_message, tags));
    });

    if (latestPrompt) {
      elements.chatThread.appendChild(
        createBubble("assistant", latestPrompt, ["next prompt", payload.current_target_slot])
      );
    }
  }

  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

function createBubble(role, text, tags = []) {
  const article = document.createElement("article");
  article.className = `bubble ${role}`;
  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  tags.filter(Boolean).forEach((tag) => {
    const span = document.createElement("span");
    span.className = "bubble-tag";
    span.textContent = tag;
    meta.appendChild(span);
  });
  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;
  article.append(meta, body);
  return article;
}

function renderSummary(summary, payload) {
  elements.summaryCards.innerHTML = "";
  const cards = [
    { label: "演算法", value: summary.algorithm.toUpperCase() },
    { label: "總輪數", value: String(summary.total_turns) },
    { label: "平均相似度", value: String(summary.average_similarity) },
    { label: "轉場次數", value: String(summary.transitions_used) },
  ];

  cards.forEach((item) => {
    const block = document.createElement("article");
    block.className = "summary-item";
    block.innerHTML = `<strong>${item.value}</strong><div>${item.label}</div>`;
    elements.summaryCards.appendChild(block);
  });

  elements.slotProgress.innerHTML = "";
  Object.entries(summary.slot_completion).forEach(([slotName, info]) => {
    const card = document.createElement("article");
    card.className = "slot-card";
    const items = info.filled_items.length ? info.filled_items.join("、") : "尚未蒐集";
    card.innerHTML = `
      <div class="slot-head">
        <span>${slotName}</span>
        <span>${formatPercent(info.completion_ratio)}</span>
      </div>
      <div class="slot-bar">
        <div class="slot-fill" style="width: ${formatPercent(info.completion_ratio)}"></div>
      </div>
      <div class="slot-meta">${items}</div>
    `;
    elements.slotProgress.appendChild(card);
  });

  renderChipList(elements.concernList, summary.concerns, "is-warning", "目前沒有顯著高風險訊號");
  renderChipList(elements.focusList, summary.next_focus_slots, "is-muted", "目前四大槽位都已有資料");
  elements.summaryMarkdown.textContent = payload.summary_markdown;
}

function renderChipList(root, items, extraClass, fallbackText) {
  root.innerHTML = "";
  const list = items && items.length ? items : [fallbackText];
  list.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = `chip ${extraClass}`;
    chip.textContent = item;
    root.appendChild(chip);
  });
}

async function createSession() {
  renderEmptyState();
  const payload = await requestJson("/api/session", { algorithm: state.algorithm });
  state.sessionId = payload.session_id;
  renderChat(payload);
  renderSummary(payload.summary, payload);
  speak(payload.latest_assistant_message);
}

async function sendMessage() {
  const message = elements.messageInput.value.trim();
  if (!message || !state.sessionId) {
    return;
  }

  elements.sendButton.disabled = true;
  try {
    const payload = await requestJson("/api/chat", {
      session_id: state.sessionId,
      message,
    });
    elements.messageInput.value = "";
    renderChat(payload);
    renderSummary(payload.summary, payload);
    speak(payload.latest_assistant_message);
  } catch (error) {
    setVoiceStatus(`送出失敗：${error.message}`);
  } finally {
    elements.sendButton.disabled = false;
  }
}

async function resetSession() {
  const payload = await requestJson("/api/reset", { algorithm: state.algorithm });
  state.sessionId = payload.session_id;
  renderChat(payload);
  renderSummary(payload.summary, payload);
  speak(payload.latest_assistant_message);
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    elements.listenButton.disabled = true;
    setVoiceStatus("此瀏覽器不支援語音輸入");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "zh-TW";
  recognition.interimResults = false;
  recognition.continuous = false;
  recognition.onstart = () => {
    state.listening = true;
    elements.listenButton.classList.add("is-active");
    setVoiceStatus("正在聽取語音...");
  };
  recognition.onend = () => {
    state.listening = false;
    elements.listenButton.classList.remove("is-active");
    setVoiceStatus("瀏覽器語音待命");
  };
  recognition.onerror = (event) => {
    setVoiceStatus(`語音輸入失敗：${event.error}`);
  };
  recognition.onresult = (event) => {
    const transcript = event.results?.[0]?.[0]?.transcript || "";
    elements.messageInput.value = transcript;
    elements.messageInput.focus();
  };
  state.recognition = recognition;
}

function toggleSpeechOutput() {
  state.speechEnabled = !state.speechEnabled;
  elements.speakToggle.textContent = state.speechEnabled ? "語音朗讀開" : "語音朗讀關";
  elements.speakToggle.classList.toggle("is-active", state.speechEnabled);
  if (!state.speechEnabled && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

function handleListen() {
  if (!state.recognition) {
    return;
  }
  if (state.listening) {
    state.recognition.stop();
  } else {
    state.recognition.start();
  }
}

function bindEvents() {
  elements.sendButton.addEventListener("click", sendMessage);
  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  elements.algorithmSelect.addEventListener("change", async (event) => {
    state.algorithm = event.target.value;
    await resetSession();
  });
  elements.resetButton.addEventListener("click", resetSession);
  elements.listenButton.addEventListener("click", handleListen);
  elements.speakToggle.addEventListener("click", toggleSpeechOutput);
}

async function init() {
  bindEvents();
  initSpeechRecognition();
  await createSession();
}

init().catch((error) => {
  setVoiceStatus(`初始化失敗：${error.message}`);
});
