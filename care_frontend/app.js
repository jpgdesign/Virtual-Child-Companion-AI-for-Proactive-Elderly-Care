const state = {
  sessionId: null,
  algorithm: "dqn",
  speechEnabled: true,
  speechRecognitionSupported: true,
  recognition: null,
  listening: false,
  lastPayload: null,
};

const elements = {
  algorithmSelect: document.getElementById("algorithm-select"),
  composerForm: document.getElementById("composer-form"),
  resetButton: document.getElementById("reset-button"),
  statusBand: document.getElementById("status-band"),
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

  const rawText = await response.text();
  let data = {};

  try {
    data = rawText ? JSON.parse(rawText) : {};
  } catch {
    data = { error: rawText || "Request failed" };
  }

  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }

  return data;
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function formatMetric(value) {
  const numeric = Number(value || 0);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "0.00";
}

function toAlgorithmLabel(name) {
  return name === "q_learning" ? "Q-learning" : String(name || "dqn").toUpperCase();
}

function getVoiceModeLabel() {
  if (state.listening) {
    return "收音中";
  }
  return state.speechEnabled ? "播報開啟" : "播報關閉";
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
  renderStatusBand(state.lastPayload);
}

function renderEmptyState() {
  elements.chatThread.innerHTML = `
    <div class="empty-state">
      <strong>正在建立新的陪伴對話 session</strong>
      <p>系統會先產生開場提示，接著你可以用文字或語音模擬長者回覆。</p>
    </div>
  `;
}

function createBubble(role, text, tags = []) {
  const article = document.createElement("article");
  article.className = `bubble ${role}`;

  const meta = document.createElement("div");
  meta.className = "bubble-meta";

  const roleChip = document.createElement("span");
  roleChip.className = "bubble-tag";
  roleChip.textContent = role === "assistant" ? "系統回覆" : "長者回覆";
  meta.appendChild(roleChip);

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

function renderChat(payload) {
  const turns = Array.isArray(payload?.turns) ? payload.turns : [];
  const latestPrompt = payload?.latest_assistant_message;
  const currentTarget = payload?.current_target_slot || "待分配槽位";

  elements.chatThread.innerHTML = "";

  if (turns.length === 0 && latestPrompt) {
    elements.chatThread.appendChild(createBubble("assistant", latestPrompt, ["開場提示", currentTarget]));
  } else {
    turns.forEach((turn) => {
      elements.chatThread.appendChild(
        createBubble("assistant", turn.assistant_message, [`劇本 ${turn.script_id}`, `目標 ${turn.target_slot}`])
      );

      const tags = [`偏離等級 ${turn.deviation_level}`];
      if (turn.transition_used) {
        tags.push("已觸發轉場");
      }

      elements.chatThread.appendChild(createBubble("user", turn.elder_message, tags));
    });

    if (latestPrompt) {
      elements.chatThread.appendChild(createBubble("assistant", latestPrompt, ["下一個提示", currentTarget]));
    }
  }

  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

function renderStatusBand(payload) {
  const summary = payload?.summary || {};
  const nextFocus = Array.isArray(summary.next_focus_slots) && summary.next_focus_slots.length
    ? summary.next_focus_slots[0]
    : "等待對話資料";

  const cards = [
    {
      label: "策略模式",
      value: toAlgorithmLabel(payload?.algorithm || state.algorithm),
      detail: "可在 DQN 與 Q-learning 間切換",
      emphasis: false,
    },
    {
      label: "目前焦點",
      value: payload?.current_target_slot || "尚未建立 session",
      detail: `下一步建議：${nextFocus}`,
      emphasis: true,
    },
    {
      label: "對話進度",
      value: `${summary.total_turns || 0} 輪`,
      detail: `偏題轉場 ${summary.transitions_used || 0} 次`,
      emphasis: false,
    },
    {
      label: "語音狀態",
      value: getVoiceModeLabel(),
      detail: elements.voiceStatus.textContent || "語音功能就緒",
      emphasis: false,
    },
  ];

  elements.statusBand.innerHTML = "";

  cards.forEach((item) => {
    const card = document.createElement("article");
    card.className = `status-card${item.emphasis ? " is-emphasis" : ""}`;
    card.innerHTML = `
      <span class="status-label">${item.label}</span>
      <span class="status-value">${item.value}</span>
      <span class="status-detail">${item.detail}</span>
    `;
    elements.statusBand.appendChild(card);
  });
}

function renderSummary(summary, payload) {
  const safeSummary = summary || {};

  elements.summaryCards.innerHTML = "";
  const cards = [
    { label: "演算法", value: toAlgorithmLabel(safeSummary.algorithm || state.algorithm) },
    { label: "對話輪數", value: String(safeSummary.total_turns || 0) },
    { label: "平均相似度", value: formatMetric(safeSummary.average_similarity) },
    { label: "平均偏離", value: formatMetric(safeSummary.average_deviation) },
    { label: "轉場次數", value: String(safeSummary.transitions_used || 0) },
  ];

  cards.forEach((item) => {
    const block = document.createElement("article");
    block.className = "summary-item";
    block.innerHTML = `<strong>${item.value}</strong><span>${item.label}</span>`;
    elements.summaryCards.appendChild(block);
  });

  elements.slotProgress.innerHTML = "";
  const slotCompletion = safeSummary.slot_completion || {};
  Object.entries(slotCompletion).forEach(([slotName, info]) => {
    const card = document.createElement("article");
    card.className = "slot-card";
    const filledItems = Array.isArray(info.filled_items) && info.filled_items.length
      ? info.filled_items.join("、")
      : "尚未收集到具體項目";

    card.innerHTML = `
      <div class="slot-head">
        <span>${slotName}</span>
        <span>${formatPercent(info.completion_ratio)}</span>
      </div>
      <div class="slot-bar">
        <div class="slot-fill" style="width: ${formatPercent(info.completion_ratio)}"></div>
      </div>
      <div class="slot-meta">${filledItems}</div>
    `;
    elements.slotProgress.appendChild(card);
  });

  renderChipList(
    elements.concernList,
    safeSummary.concerns,
    "is-warning",
    "目前對話尚未偵測到明顯高風險訊號"
  );
  renderChipList(
    elements.focusList,
    safeSummary.next_focus_slots,
    "is-muted",
    "目前四大槽位皆已有資料，可改追蹤趨勢變化"
  );

  elements.summaryMarkdown.textContent = payload?.summary_markdown || "尚無摘要內容。";
}

function renderChipList(root, items, extraClass, fallbackText) {
  root.innerHTML = "";
  const list = Array.isArray(items) && items.length ? items : [fallbackText];

  list.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = `chip ${extraClass}`;
    chip.textContent = item;
    root.appendChild(chip);
  });
}

function applyPayload(payload) {
  state.sessionId = payload.session_id || state.sessionId;
  state.lastPayload = payload;
  renderStatusBand(payload);
  renderChat(payload);
  renderSummary(payload.summary, payload);
}

function setSendingState(isSending) {
  elements.sendButton.disabled = isSending;
  elements.sendButton.textContent = isSending ? "送出中..." : "送出訊息";
}

async function createSession() {
  renderEmptyState();
  const payload = await requestJson("/api/session", { algorithm: state.algorithm });
  applyPayload(payload);
  speak(payload.latest_assistant_message);
}

async function sendMessage() {
  const message = elements.messageInput.value.trim();
  if (!message || !state.sessionId) {
    return;
  }

  setSendingState(true);
  try {
    const payload = await requestJson("/api/chat", {
      session_id: state.sessionId,
      message,
    });
    elements.messageInput.value = "";
    applyPayload(payload);
    setVoiceStatus("已更新回應與摘要");
    speak(payload.latest_assistant_message);
  } catch (error) {
    setVoiceStatus(`送出失敗：${error.message}`);
  } finally {
    setSendingState(false);
  }
}

async function resetSession() {
  renderEmptyState();
  const payload = await requestJson("/api/reset", { algorithm: state.algorithm });
  applyPayload(payload);
  setVoiceStatus("已建立新的對話 session");
  speak(payload.latest_assistant_message);
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    state.speechRecognitionSupported = false;
    elements.listenButton.disabled = true;
    setVoiceStatus("此瀏覽器不支援語音輸入，請改用文字輸入");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "zh-TW";
  recognition.interimResults = false;
  recognition.continuous = false;

  recognition.onstart = () => {
    state.listening = true;
    elements.listenButton.classList.add("is-active");
    setVoiceStatus("正在聆聽語音輸入...");
  };

  recognition.onend = () => {
    state.listening = false;
    elements.listenButton.classList.remove("is-active");
    setVoiceStatus("語音輸入待命中");
  };

  recognition.onerror = (event) => {
    setVoiceStatus(`語音輸入失敗：${event.error}`);
  };

  recognition.onresult = (event) => {
    const transcript = event.results?.[0]?.[0]?.transcript || "";
    elements.messageInput.value = transcript;
    elements.messageInput.focus();
    setVoiceStatus("已帶入語音辨識結果");
  };

  state.recognition = recognition;
}

function toggleSpeechOutput() {
  state.speechEnabled = !state.speechEnabled;
  elements.speakToggle.textContent = state.speechEnabled ? "語音播報已開" : "語音播報已關";
  elements.speakToggle.classList.toggle("is-active", state.speechEnabled);

  if (!state.speechEnabled && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  setVoiceStatus(state.speechEnabled ? "語音播報已開啟" : "語音播報已關閉");
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
  elements.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });

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
  elements.algorithmSelect.value = state.algorithm;
  bindEvents();
  initSpeechRecognition();
  await createSession();
}

init().catch((error) => {
  setVoiceStatus(`初始化失敗：${error.message}`);
});
