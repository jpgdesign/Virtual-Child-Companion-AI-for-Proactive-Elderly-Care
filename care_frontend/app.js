const fallbackPresets = {
  qwen3_ollama: {
    provider: "ollama",
    model: "qwen3:8b",
  },
  qwen35_lmstudio: {
    provider: "openai",
    model: "qwen3.5-9b-claude",
  },
};

const state = {
  sessionId: null,
  algorithm: "dqn",
  llmEnabled: true,
  analysisPreset: "qwen35_lmstudio",
  generationPreset: "qwen35_lmstudio",
  availablePresets: { ...fallbackPresets },
  speechEnabled: true,
  recognition: null,
  listening: false,
  lastPayload: null,
  pollTimer: null,
  backgroundPolling: false,
};

const elements = {
  algorithmSelect: document.getElementById("algorithm-select"),
  analysisPresetSelect: document.getElementById("analysis-preset-select"),
  generationPresetSelect: document.getElementById("generation-preset-select"),
  llmEnabledCheckbox: document.getElementById("llm-enabled-checkbox"),
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
  llmOverview: document.getElementById("llm-overview"),
  emotionPanel: document.getElementById("emotion-panel"),
  analysisSummary: document.getElementById("analysis-summary"),
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

function getPresetLabel(presetId) {
  const preset = state.availablePresets[presetId] || fallbackPresets[presetId];
  if (!preset) {
    return presetId;
  }
  return `${preset.model} (${preset.provider})`;
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
      <p>系統會先用 RL 決定目標槽位，再快速回覆，背景補上 LLM 分析與自然化調整。</p>
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
      const assistantTags = [`劇本 ${turn.script_id}`, `目標 ${turn.target_slot}`];
      if (turn.generated_by_llm) {
        assistantTags.push("LLM 輔助");
      }
      elements.chatThread.appendChild(createBubble("assistant", turn.assistant_message, assistantTags));

      const tags = [`偏離等級 ${turn.deviation_level}`];
      if (turn.transition_used) {
        tags.push("已轉場");
      }
      if (turn.emotion_label) {
        tags.push(`情緒 ${turn.emotion_label}`);
      }
      elements.chatThread.appendChild(createBubble("user", turn.elder_message, tags));
    });

    if (latestPrompt) {
      const latestTags = ["快速回覆", currentTarget];
      if (payload?.background_processing) {
        latestTags.push("背景優化中");
      }
      elements.chatThread.appendChild(createBubble("assistant", latestPrompt, latestTags));
    }
  }

  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

function renderStatusBand(payload) {
  const summary = payload?.summary || {};
  const llmStatus = summary.llm_status || payload?.llm_status || {};
  const nextFocus = Array.isArray(summary.next_focus_slots) && summary.next_focus_slots.length
    ? summary.next_focus_slots[0]
    : "等待對話資料";

  const cards = [
    {
      label: "策略模式",
      value: toAlgorithmLabel(payload?.algorithm || state.algorithm),
      detail: "由 RL 決定下一份劇本",
      emphasis: false,
    },
    {
      label: "目前焦點",
      value: payload?.current_target_slot || "尚未建立 session",
      detail: `下一步建議：${nextFocus}`,
      emphasis: true,
    },
    {
      label: "回覆模式",
      value: payload?.background_processing ? "極速模式" : "分析完成",
      detail: payload?.background_processing
        ? "先秒回，背景補上分析與自然化"
        : "目前顯示的是最新整理後版本",
      emphasis: false,
    },
    {
      label: "模型路徑",
      value: llmStatus.enabled ? "RL + LLM" : "純規則",
      detail: llmStatus.enabled
        ? `${getPresetLabel(state.analysisPreset)} / ${getPresetLabel(state.generationPreset)}`
        : "目前未啟用 LLM",
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

function renderLLMOverview(summary, payload) {
  const llmStatus = summary.llm_status || payload?.llm_status || {};
  const analysis = payload?.latest_analysis || {};
  const emotion = summary.latest_emotion || analysis.emotion || {};
  const statusLabel = analysis.status || (payload?.background_processing ? "pending" : "idle");

  const entries = [
    { label: "分析狀態", value: statusLabel },
    { label: "分析模型", value: llmStatus.analysis?.model || getPresetLabel(state.analysisPreset) },
    { label: "生成模型", value: llmStatus.generation?.model || getPresetLabel(state.generationPreset) },
    { label: "最新情緒", value: emotion.label || "尚未分析" },
  ];

  elements.llmOverview.innerHTML = "";
  entries.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip is-muted";
    chip.textContent = `${item.label}：${item.value}`;
    elements.llmOverview.appendChild(chip);
  });
}

function renderEmotionPanel(summary, payload) {
  const analysis = payload?.latest_analysis || {};
  const emotion = summary.latest_emotion || analysis.emotion || {};
  const concerns = Array.isArray(analysis.concerns) ? analysis.concerns : [];

  const items = [
    { label: "情緒", value: emotion.label || "未分析" },
    { label: "強度", value: formatMetric(emotion.intensity) },
    { label: "偏離", value: String(analysis.deviation_level ?? summary.average_deviation ?? 0) },
    { label: "提醒", value: concerns.length ? `${concerns.length} 項` : "無" },
  ];

  elements.emotionPanel.innerHTML = "";
  items.forEach((item) => {
    const block = document.createElement("article");
    block.className = "status-card";
    block.innerHTML = `
      <span class="status-label">${item.label}</span>
      <span class="status-value">${item.value}</span>
    `;
    elements.emotionPanel.appendChild(block);
  });

  elements.analysisSummary.textContent =
    analysis.summary || summary.latest_analysis_summary || "背景分析尚未完成。";
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

  renderLLMOverview(safeSummary, payload);
  renderEmotionPanel(safeSummary, payload);

  elements.slotProgress.innerHTML = "";
  const slotCompletion = safeSummary.slot_completion || {};
  Object.entries(slotCompletion).forEach(([slotName, info]) => {
    const filledItems = Array.isArray(info.filled_items) && info.filled_items.length
      ? info.filled_items.join("、")
      : "尚未收集到具體項目";
    const valueNotes = Object.entries(info.value_notes || {})
      .map(([item, values]) => `${item}：${Array.isArray(values) ? values.join(" / ") : ""}`)
      .filter(Boolean)
      .join("；");

    const card = document.createElement("article");
    card.className = "slot-card";
    card.innerHTML = `
      <div class="slot-head">
        <span>${slotName}</span>
        <span>${formatPercent(info.completion_ratio)}</span>
      </div>
      <div class="slot-bar">
        <div class="slot-fill" style="width: ${formatPercent(info.completion_ratio)}"></div>
      </div>
      <div class="slot-meta">${filledItems}</div>
      ${valueNotes ? `<div class="slot-meta">${valueNotes}</div>` : ""}
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

function syncPresetState(payload) {
  if (payload?.available_model_presets) {
    state.availablePresets = payload.available_model_presets;
  }
  if (payload?.llm_status?.analysis?.preset) {
    state.analysisPreset = payload.llm_status.analysis.preset;
  }
  if (payload?.llm_status?.generation?.preset) {
    state.generationPreset = payload.llm_status.generation.preset;
  }
  if (payload?.llm_status?.enabled !== undefined) {
    state.llmEnabled = Boolean(payload.llm_status.enabled);
  }
  populatePresetOptions();
}

function populatePresetOptions() {
  const presets = state.availablePresets || fallbackPresets;
  const optionsHtml = Object.keys(presets)
    .map((key) => `<option value="${key}">${getPresetLabel(key)}</option>`)
    .join("");
  elements.analysisPresetSelect.innerHTML = optionsHtml;
  elements.generationPresetSelect.innerHTML = optionsHtml;
  elements.analysisPresetSelect.value = state.analysisPreset;
  elements.generationPresetSelect.value = state.generationPreset;
  elements.llmEnabledCheckbox.checked = state.llmEnabled;
}

function buildSessionConfig() {
  return {
    algorithm: state.algorithm,
    llm_enabled: state.llmEnabled,
    analysis_preset: state.analysisPreset,
    generation_preset: state.generationPreset,
  };
}

function stopBackgroundPolling() {
  if (state.pollTimer) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  state.backgroundPolling = false;
}

function scheduleBackgroundPolling() {
  stopBackgroundPolling();
  if (!state.sessionId) {
    return;
  }
  state.backgroundPolling = true;
  state.pollTimer = window.setTimeout(async () => {
    try {
      const payload = await requestJson("/api/session_state", { session_id: state.sessionId });
      const wasProcessing = Boolean(state.lastPayload?.background_processing);
      applyPayload(payload);
      if (payload.background_processing) {
        scheduleBackgroundPolling();
      } else {
        stopBackgroundPolling();
        if (wasProcessing) {
          setVoiceStatus("背景分析已完成，摘要已更新");
        }
      }
    } catch (error) {
      stopBackgroundPolling();
      setVoiceStatus(`背景更新失敗：${error.message}`);
    }
  }, 1500);
}

function applyPayload(payload) {
  state.sessionId = payload.session_id || state.sessionId;
  state.lastPayload = payload;
  syncPresetState(payload);
  renderStatusBand(payload);
  renderChat(payload);
  renderSummary(payload.summary, payload);
}

function setSendingState(isSending) {
  elements.sendButton.disabled = isSending;
  elements.sendButton.textContent = isSending ? "送出中..." : "送出訊息";
}

async function createSession() {
  stopBackgroundPolling();
  renderEmptyState();
  const payload = await requestJson("/api/session", buildSessionConfig());
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
    setVoiceStatus(
      payload.background_processing
        ? "已快速回覆，背景正在補情緒分析與自然化版本"
        : "已更新回應與摘要"
    );
    speak(payload.latest_assistant_message);
    if (payload.background_processing) {
      scheduleBackgroundPolling();
    } else {
      stopBackgroundPolling();
    }
  } catch (error) {
    setVoiceStatus(`送出失敗：${error.message}`);
  } finally {
    setSendingState(false);
  }
}

async function resetSession() {
  stopBackgroundPolling();
  renderEmptyState();
  const payload = await requestJson("/api/reset", buildSessionConfig());
  applyPayload(payload);
  setVoiceStatus("已建立新的對話 session");
  speak(payload.latest_assistant_message);
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
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

async function handleConfigChange() {
  state.algorithm = elements.algorithmSelect.value;
  state.analysisPreset = elements.analysisPresetSelect.value;
  state.generationPreset = elements.generationPresetSelect.value;
  state.llmEnabled = elements.llmEnabledCheckbox.checked;
  if (state.sessionId) {
    await resetSession();
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

  elements.algorithmSelect.addEventListener("change", handleConfigChange);
  elements.analysisPresetSelect.addEventListener("change", handleConfigChange);
  elements.generationPresetSelect.addEventListener("change", handleConfigChange);
  elements.llmEnabledCheckbox.addEventListener("change", handleConfigChange);
  elements.resetButton.addEventListener("click", resetSession);
  elements.listenButton.addEventListener("click", handleListen);
  elements.speakToggle.addEventListener("click", toggleSpeechOutput);
}

async function init() {
  populatePresetOptions();
  elements.algorithmSelect.value = state.algorithm;
  bindEvents();
  initSpeechRecognition();
  await createSession();
}

init().catch((error) => {
  setVoiceStatus(`初始化失敗：${error.message}`);
});
