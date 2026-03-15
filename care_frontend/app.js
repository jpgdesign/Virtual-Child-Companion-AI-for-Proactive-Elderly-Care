const state = {
  bootstrap: null,
  currentView: "landing",
  authToken: "",
  currentUser: null,
  adminToken: "",
  adminUser: null,
  sessionId: null,
  lastPayload: null,
  reportData: null,
  adminData: null,
  pollTimer: null,
  backgroundPolling: false,
  loginError: "",
  adminLoginError: "",
  adminPersonaId: "",
  adminUserDraft: null,
};

const root = document.getElementById("app-root");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function requestJson(url, payload = {}, options = {}) {
  const method = options.method || "POST";
  const fetchOptions = {
    method,
    cache: "no-store",
  };
  if (method !== "GET") {
    fetchOptions.headers = { "Content-Type": "application/json" };
    fetchOptions.body = JSON.stringify(payload);
  }
  const response = await fetch(url, fetchOptions);
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

async function requestBootstrap() {
  try {
    return await requestJson("/api/bootstrap");
  } catch (error) {
    if (!String(error.message || "").includes("Unknown endpoint")) {
      throw error;
    }

    try {
      const health = await requestJson("/api/health", {}, { method: "GET" });
      if (health?.ok) {
        throw new Error(
          "偵測到舊版後端仍在執行。請先停止目前的 Python server，再重新執行 `py care_companion_server.py --open-browser`。"
        );
      }
    } catch (healthError) {
      if (healthError instanceof Error && healthError.message !== "Unknown endpoint.") {
        throw healthError;
      }
    }

    throw error;
  }
}

function formatMetric(value) {
  const numeric = Number(value || 0);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "0.00";
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function getBootstrap() {
  return state.bootstrap || { demo_accounts: [], personas: {}, api_settings: {}, prompt_settings: {} };
}

function getCurrentPayload() {
  return state.lastPayload || {};
}

function getCurrentSummary() {
  return getCurrentPayload().summary || {};
}

function getCurrentPersona() {
  return getCurrentPayload().persona_profile || {};
}

function getAssistantName() {
  const child = getCurrentPersona().child || {};
  return child.name || "虛擬兒女";
}

function getCurrentFamilyLabel() {
  return getCurrentPersona().label || state.currentUser?.persona_label || "尚未登入家庭";
}

function getDefaultUserDraft() {
  return {
    id: "",
    username: "",
    password: "",
    display_name: "",
    role: "family",
    persona_profile_id: "daughter_teacher_mother",
    default_view: "family",
    enabled: true,
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
  if (!state.sessionId || !state.authToken) {
    return;
  }
  state.backgroundPolling = true;
  state.pollTimer = window.setTimeout(async () => {
    try {
      const payload = await requestJson("/api/session_state", {
        auth_token: state.authToken,
        session_id: state.sessionId,
      });
      state.lastPayload = payload;
      renderApp();
      if (payload.background_processing) {
        scheduleBackgroundPolling();
      } else {
        stopBackgroundPolling();
      }
    } catch (error) {
      stopBackgroundPolling();
      console.error(error);
    }
  }, 1500);
}

async function ensureConversationSession() {
  if (!state.authToken) {
    throw new Error("需要先登入。");
  }
  if (state.sessionId && state.lastPayload) {
    return;
  }
  const payload = await requestJson("/api/session", {
    auth_token: state.authToken,
  });
  state.sessionId = payload.session_id;
  state.lastPayload = payload;
}

async function loadReportData() {
  if (!state.authToken) {
    return;
  }
  state.reportData = await requestJson("/api/report", { auth_token: state.authToken });
  if (!state.lastPayload && state.reportData?.latest_session) {
    state.lastPayload = state.reportData.latest_session;
    state.sessionId = state.reportData.latest_session.session_id;
  }
}

async function loadAdminState() {
  if (!state.adminToken) {
    return;
  }
  state.bootstrap = await requestJson("/api/bootstrap");
  state.adminData = await requestJson("/api/admin/state", { auth_token: state.adminToken });
  if (!state.adminPersonaId) {
    state.adminPersonaId = Object.keys(state.adminData.personas || {})[0] || "";
  }
  if (!state.adminUserDraft) {
    state.adminUserDraft = getDefaultUserDraft();
  }
}

async function navigate(view) {
  state.currentView = view;
  if (view === "family" || view === "elder") {
    await ensureConversationSession();
  }
  if (view === "report") {
    await loadReportData();
  }
  if (view === "admin") {
    await loadAdminState();
  }
  renderApp();
}

function renderTopbar() {
  const userLabel = state.currentUser
    ? `${state.currentUser.display_name}｜${state.currentUser.role === "family" ? "家屬" : "長者"}`
    : "尚未登入";
  const adminButtonLabel = state.adminToken ? "後台管理" : "登入後台";
  return `
    <header class="topbar">
      <div class="brand-block">
        <span class="brand-kicker">Virtual Child Care Companion</span>
        <strong>虛擬子女照護陪伴系統</strong>
      </div>
      <div class="topbar-actions">
        <span class="user-pill">${escapeHtml(userLabel)}</span>
        <button class="ghost-button" data-action="go-landing">導覽首頁</button>
        ${state.currentUser ? '<button class="ghost-button" data-action="go-portal">功能入口</button>' : '<button class="ghost-button" data-action="go-login">登入系統</button>'}
        <button class="ghost-button is-admin" data-action="go-admin">${adminButtonLabel}</button>
        ${state.currentUser ? '<button class="ghost-button" data-action="logout-user">登出使用者</button>' : ""}
      </div>
    </header>
  `;
}

function renderGuideCards() {
  return `
    <div class="guide-grid">
      <article class="guide-card">
        <span class="highlight-kicker">RL 決策</span>
        <strong>DQN / Q-learning</strong>
        <p>保留可比較的對話策略選擇，下一步要問哪個槽位仍由 RL 負責。</p>
      </article>
      <article class="guide-card">
        <span class="highlight-kicker">LLM 整合</span>
        <strong>理解 + 生成</strong>
        <p>情緒分析、填槽、偏離判斷與自然回覆整合在同一套家人畫像之上。</p>
      </article>
      <article class="guide-card">
        <span class="highlight-kicker">角色模式</span>
        <strong>家屬 / 長者 / 報告</strong>
        <p>登入後依需求進入不同工作區塊，避免把所有資訊一次塞滿。</p>
      </article>
    </div>
  `;
}

function renderLandingView() {
  const personas = Object.values(getBootstrap().personas || {});
  const personaCards = personas.map((persona) => {
    const child = persona.child || {};
    const elder = persona.elder || {};
    const relationship = persona.relationship || {};
    return `
      <article class="persona-preview">
        <span class="section-label">Persona Context</span>
        <h3>${escapeHtml(persona.label || "")}</h3>
        <p>${escapeHtml(relationship.family_mapping || "")}</p>
        <div class="persona-meta">
          <span>${escapeHtml(child.name || "-")}｜${escapeHtml(child.role || "-")}</span>
          <span>${escapeHtml(elder.name || "-")}｜${escapeHtml(elder.role || "-")}</span>
        </div>
      </article>
    `;
  }).join("");

  return `
    <section class="hero hero-landing">
      <div class="hero-copy">
        <p class="eyebrow">Interactive Care Frontend</p>
        <h1>先導覽，再登入，再依角色進入對應工作區</h1>
        <p class="hero-text">
          這不是單一聊天頁，而是完整的虛擬子女產品流程。先看服務說明，再用家庭帳號登入，
          接著選擇進入家屬、長者或報告模式；右上角則可再切進後台管理畫像、提示詞、帳號與 API。
        </p>
        <div class="hero-actions">
          <button class="send-button" data-action="go-login">前往登入</button>
          <button class="ghost-button" data-action="go-portal-demo">直接看功能入口</button>
        </div>
      </div>
      <div class="hero-side">${renderGuideCards()}</div>
    </section>

    <section class="dashboard-card landing-section">
      <p class="section-label">Family Sets</p>
      <h2>三組家庭畫像</h2>
      <div class="persona-preview-grid">${personaCards}</div>
    </section>
  `;
}

function renderLoginAccounts() {
  const accounts = getBootstrap().demo_accounts || [];
  return accounts.map((account) => `
    <article class="account-card">
      <div>
        <span class="highlight-kicker">${escapeHtml(account.role === "family" ? "家屬帳號" : "長者帳號")}</span>
        <strong>${escapeHtml(account.display_name)}</strong>
        <p>${escapeHtml(account.persona_label || "")}</p>
      </div>
      <div class="account-credentials">
        <code>${escapeHtml(account.username)}</code>
        <code>${escapeHtml(account.password)}</code>
      </div>
      <button class="ghost-button" data-fill-username="${escapeHtml(account.username)}" data-fill-password="${escapeHtml(account.password)}">帶入登入</button>
    </article>
  `).join("");
}

function renderLoginView() {
  return `
    <section class="auth-layout">
      <article class="auth-card">
        <p class="section-label">Login</p>
        <h2>選擇家庭帳號登入</h2>
        <p class="panel-text">每組家庭目前提供 2 個帳號：家屬帳號與長者帳號。登入後可再選擇進入家屬、長者或報告模式。</p>
        <form id="login-form" class="auth-form">
          <label class="field">
            <span>帳號</span>
            <input id="login-username" class="text-input" autocomplete="username" />
          </label>
          <label class="field">
            <span>密碼</span>
            <input id="login-password" class="text-input" type="password" autocomplete="current-password" />
          </label>
          ${state.loginError ? `<p class="form-error">${escapeHtml(state.loginError)}</p>` : ""}
          <button class="send-button" type="submit">登入系統</button>
        </form>
      </article>

      <article class="dashboard-card">
        <p class="section-label">Demo Accounts</p>
        <h2>六組示範帳號</h2>
        <div class="account-grid">${renderLoginAccounts()}</div>
      </article>
    </section>
  `;
}

function renderPortalView() {
  const user = state.currentUser || {};
  return `
    <section class="hero portal-hero">
      <div class="hero-copy">
        <p class="eyebrow">Role Entrance</p>
        <h1>${escapeHtml(user.display_name || "已登入使用者")}</h1>
        <p class="hero-text">目前登入家庭：${escapeHtml(user.persona_label || "")}。請選擇接下來要進入的工作模式。</p>
      </div>
      <div class="portal-grid">
        <button class="portal-card" data-action="enter-family">
          <span class="highlight-kicker">1. 家屬</span>
          <strong>對話 + 填槽 + 摘要</strong>
          <p>進入完整照護工作台，包含互動摘要、家庭關係畫像、分析結果與家屬摘要。</p>
        </button>
        <button class="portal-card" data-action="enter-elder">
          <span class="highlight-kicker">2. 長者</span>
          <strong>純聊天介面</strong>
          <p>只顯示陪伴對話區，讓長者看到的畫面更單純、更像在和兒女聊天。</p>
        </button>
        <button class="portal-card" data-action="enter-report">
          <span class="highlight-kicker">3. 報告</span>
          <strong>視覺儀表板</strong>
          <p>把對話紀錄、填槽與照護提醒整合成一頁式 dashboard，方便追蹤趨勢。</p>
        </button>
      </div>
    </section>
  `;
}

function renderStatusCards(payload) {
  const summary = payload.summary || {};
  const llmStatus = payload.llm_status || {};
  const nextFocus = Array.isArray(summary.next_focus_slots) && summary.next_focus_slots.length
    ? summary.next_focus_slots[0]
    : "等待新資料";
  const cards = [
    {
      label: "目前家庭",
      value: getCurrentFamilyLabel(),
      detail: payload.user?.display_name || "",
    },
    {
      label: "策略模式",
      value: String(payload.algorithm || "dqn").toUpperCase(),
      detail: `下一步優先焦點：${nextFocus}`,
    },
    {
      label: "回覆模式",
      value: payload.background_processing ? "極速回覆中" : "分析完成",
      detail: payload.background_processing ? "先回應，再背景補分析與自然化" : "目前顯示最新完成版本",
    },
    {
      label: "模型路徑",
      value: llmStatus.enabled ? "RL + LLM" : "純規則",
      detail: llmStatus.analysis?.model || "",
    },
  ];

  return cards.map((item, index) => `
    <article class="status-card${index === 1 ? " is-emphasis" : ""}">
      <span class="status-label">${escapeHtml(item.label)}</span>
      <span class="status-value">${escapeHtml(item.value)}</span>
      <span class="status-detail">${escapeHtml(item.detail)}</span>
    </article>
  `).join("");
}

function renderChatThread(payload, compact = false) {
  const turns = Array.isArray(payload.turns) ? payload.turns : [];
  const latestPrompt = payload.latest_assistant_message || "";
  const assistantName = getAssistantName();
  const bubbles = [];

  if (turns.length === 0 && latestPrompt) {
    bubbles.push(`
      <article class="bubble assistant">
        <div class="bubble-meta">
          <span class="bubble-tag">${escapeHtml(assistantName)}</span>
          <span class="bubble-tag">開場</span>
        </div>
        <div class="bubble-body">${escapeHtml(latestPrompt)}</div>
      </article>
    `);
  }

  turns.forEach((turn) => {
    bubbles.push(`
      <article class="bubble assistant">
        <div class="bubble-meta">
          <span class="bubble-tag">${escapeHtml(assistantName)}</span>
          <span class="bubble-tag">劇本 ${escapeHtml(turn.script_id)}</span>
          <span class="bubble-tag">${escapeHtml(turn.target_slot)}</span>
        </div>
        <div class="bubble-body">${escapeHtml(turn.assistant_message)}</div>
      </article>
      <article class="bubble user">
        <div class="bubble-meta">
          <span class="bubble-tag">長者回覆</span>
          <span class="bubble-tag">偏離 ${escapeHtml(turn.deviation_level)}</span>
        </div>
        <div class="bubble-body">${escapeHtml(turn.elder_message)}</div>
      </article>
    `);
  });

  if (turns.length > 0 && latestPrompt) {
    bubbles.push(`
      <article class="bubble assistant">
        <div class="bubble-meta">
          <span class="bubble-tag">${escapeHtml(assistantName)}</span>
          <span class="bubble-tag">${payload.background_processing ? "背景優化中" : "最新回覆"}</span>
        </div>
        <div class="bubble-body">${escapeHtml(latestPrompt)}</div>
      </article>
    `);
  }

  return `
    <div class="chat-thread${compact ? " is-compact" : ""}">
      ${bubbles.join("") || `
        <div class="empty-state">
          <strong>正在建立新的陪伴對話</strong>
          <p>登入後會依據家庭帳號自動帶入對應的兒女人設與長者資料。</p>
        </div>
      `}
    </div>
  `;
}

function renderComposer(buttonLabel = "送出訊息") {
  return `
    <form id="chat-form" class="composer">
      <label class="composer-field">
        <span>輸入長者回覆</span>
        <textarea id="chat-input" rows="4" placeholder="例如：我今天早上有去公園走走一下，現在覺得還好。"></textarea>
      </label>
      <div class="composer-actions">
        <p class="composer-note">系統會先快速回應，再背景補分析與自然化版本。</p>
        <button class="send-button" type="submit">${buttonLabel}</button>
      </div>
    </form>
  `;
}

function renderPersonaSection(persona) {
  const child = persona.child || {};
  const elder = persona.elder || {};
  const relationship = persona.relationship || {};
  return `
    <section class="dashboard-card">
      <p class="section-label">Persona Context</p>
      <h2>家庭關係畫像</h2>
      <div class="slot-stack">
        <article class="slot-card">
          <div class="slot-head"><span>${escapeHtml(persona.label || "尚未選擇家庭畫像")}</span></div>
          <p class="slot-notes">${escapeHtml(relationship.family_mapping || "尚未建立家庭關係資料。")}</p>
        </article>
        <article class="slot-card">
          <div class="slot-head"><span>虛擬兒女：${escapeHtml(child.name || "-")}</span></div>
          <p class="slot-notes">${escapeHtml([child.role_detail || child.role || "", child.occupation || "", ...(child.personality || []).slice(0, 3)].filter(Boolean).join("｜")) || "尚無資料"}</p>
        </article>
        <article class="slot-card">
          <div class="slot-head"><span>長者：${escapeHtml(elder.name || "-")}</span></div>
          <p class="slot-notes">${escapeHtml([elder.role || "", elder.living_status || "", ...(elder.health_notes || []).slice(0, 3)].filter(Boolean).join("｜")) || "尚無資料"}</p>
        </article>
        <article class="slot-card">
          <div class="slot-head"><span>互動準則</span></div>
          <p class="slot-notes">${escapeHtml(relationship.guidance_style || "先接住話題，再慢慢引導。")}</p>
        </article>
      </div>
    </section>
  `;
}

function renderSummarySection(summary, payload) {
  const cards = [
    { label: "演算法", value: String(summary.algorithm || "dqn").toUpperCase() },
    { label: "對話輪數", value: String(summary.total_turns || 0) },
    { label: "平均相似度", value: formatMetric(summary.average_similarity) },
    { label: "平均偏離", value: formatMetric(summary.average_deviation) },
    { label: "轉場次數", value: String(summary.transitions_used || 0) },
    { label: "最新情緒", value: summary.latest_emotion?.label || "尚未分析" },
  ];

  const emotionItems = [
    { label: "情緒", value: summary.latest_emotion?.label || "未分析" },
    { label: "強度", value: formatMetric(summary.latest_emotion?.intensity) },
    { label: "模式", value: payload.background_processing ? "背景分析中" : "已完成" },
    { label: "提醒", value: (summary.concerns || []).length ? `${summary.concerns.length} 項` : "無" },
  ];

  const slotCards = Object.entries(summary.slot_completion || {}).map(([slotName, info]) => {
    const valueNotes = Object.entries(info.value_notes || {})
      .map(([item, values]) => `${item}：${Array.isArray(values) ? values.join(" / ") : ""}`)
      .join("；");
    return `
      <article class="slot-card">
        <div class="slot-head">
          <span>${escapeHtml(slotName)}</span>
          <span>${formatPercent(info.completion_ratio)}</span>
        </div>
        <div class="slot-bar"><div class="slot-fill" style="width: ${formatPercent(info.completion_ratio)}"></div></div>
        <p class="slot-meta">已蒐集：${escapeHtml((info.filled_items || []).join("、") || "尚未蒐集")}</p>
        <p class="slot-notes">${escapeHtml(valueNotes || "尚無值摘要")}</p>
      </article>
    `;
  }).join("");

  return `
    <aside class="dashboard">
      <section class="dashboard-card">
        <p class="section-label">Session Snapshot</p>
        <h2>互動摘要</h2>
        <div class="summary-grid">
          ${cards.map((item) => `
            <article class="summary-item">
              <strong>${escapeHtml(item.value)}</strong>
              <span>${escapeHtml(item.label)}</span>
            </article>
          `).join("")}
        </div>
      </section>

      ${renderPersonaSection(payload.persona_profile || {})}

      <section class="dashboard-card">
        <p class="section-label">LLM Insight</p>
        <h2>分析結果</h2>
        <div class="emotion-panel">
          ${emotionItems.map((item) => `
            <article class="status-card">
              <span class="status-label">${escapeHtml(item.label)}</span>
              <span class="status-value">${escapeHtml(item.value)}</span>
            </article>
          `).join("")}
        </div>
        <p class="analysis-note">${escapeHtml(payload.latest_analysis?.summary || summary.latest_analysis_summary || "背景分析尚未完成。")}</p>
      </section>

      <section class="dashboard-card">
        <p class="section-label">Slot Progress</p>
        <h2>四大槽位進度</h2>
        <div class="slot-stack">${slotCards}</div>
      </section>

      <section class="dashboard-card">
        <p class="section-label">Care Alerts</p>
        <h2>照護提醒</h2>
        <div class="chip-list">
          ${(summary.concerns || ["目前沒有高風險提醒"]).map((item) => `<span class="chip is-warning">${escapeHtml(item)}</span>`).join("")}
        </div>
        <h3 class="subheading">下一步建議焦點</h3>
        <div class="chip-list">
          ${(summary.next_focus_slots || ["目前四大槽位皆已有資料"]).map((item) => `<span class="chip is-muted">${escapeHtml(item)}</span>`).join("")}
        </div>
      </section>

      <section class="dashboard-card">
        <p class="section-label">Caregiver Summary</p>
        <h2>家屬摘要</h2>
        <pre class="summary-markdown">${escapeHtml(payload.summary_markdown || "")}</pre>
      </section>
    </aside>
  `;
}

function renderFamilyView() {
  const payload = getCurrentPayload();
  const summary = getCurrentSummary();
  return `
    <section class="hero module-hero">
      <div class="hero-copy">
        <p class="eyebrow">Family Workspace</p>
        <h1>家屬模式</h1>
        <p class="hero-text">保留現在的對話框與填槽流程，並整合互動摘要、家庭關係畫像、分析結果、四大槽位進度、照護提醒與家屬摘要。</p>
      </div>
      <div class="hero-side">${renderGuideCards()}</div>
    </section>

    <div class="status-band">${renderStatusCards(payload)}</div>

    <main class="layout">
      <section class="chat-panel">
        <div class="panel-head">
          <div>
            <p class="section-label">Conversation Runtime</p>
            <h2>家屬互動工作區</h2>
            <p class="panel-text">這裡是家屬看到的完整視角，包含即時對話、背景分析與照護摘要。</p>
          </div>
        </div>
        ${renderChatThread(payload)}
        ${renderComposer("送出家屬訊息")}
      </section>
      ${renderSummarySection(summary, payload)}
    </main>
  `;
}

function renderElderView() {
  const payload = getCurrentPayload();
  return `
    <section class="hero module-hero">
      <div class="hero-copy">
        <p class="eyebrow">Elder Mode</p>
        <h1>長者模式</h1>
        <p class="hero-text">只保留聊天介面，畫面更簡潔，讓長者感受到是在和熟悉的兒女對話。</p>
      </div>
      <div class="hero-side elder-highlight">
        <article class="guide-card">
          <span class="highlight-kicker">目前家庭</span>
          <strong>${escapeHtml(getCurrentFamilyLabel())}</strong>
          <p>${escapeHtml(getCurrentPersona().relationship?.guidance_style || "先接住話題，再慢慢引導。")}</p>
        </article>
      </div>
    </section>
    <section class="chat-panel chat-panel-single">
      ${renderChatThread(payload)}
      ${renderComposer("送出長者訊息")}
    </section>
  `;
}

function renderReportRecordList(records) {
  return records.map((record) => `
    <article class="report-record">
      <div>
        <strong>${escapeHtml(record.persona_label || record.persona_profile_id || "")}</strong>
        <p>${escapeHtml(record.display_name || record.username || "")}</p>
      </div>
      <div class="report-record-meta">
        <span>${escapeHtml(record.updated_at || "")}</span>
        <span>${escapeHtml(String(record.summary?.total_turns || 0))} 輪</span>
      </div>
    </article>
  `).join("");
}

function renderReportView() {
  const report = state.reportData || { records: [], latest_session: null };
  const payload = report.latest_session || getCurrentPayload();
  const summary = payload.summary || {};
  const turns = payload.turns || [];
  return `
    <section class="hero module-hero">
      <div class="hero-copy">
        <p class="eyebrow">Report Dashboard</p>
        <h1>報告模式</h1>
        <p class="hero-text">把對話紀錄、填槽進度、家庭畫像、分析結果和照護提醒整合成視覺儀表板，方便快速巡檢。</p>
      </div>
      <div class="hero-side">
        <article class="guide-card">
          <span class="highlight-kicker">家庭帳號</span>
          <strong>${escapeHtml(getCurrentFamilyLabel())}</strong>
          <p>${escapeHtml(getCurrentPersona().relationship?.family_mapping || "")}</p>
        </article>
      </div>
    </section>

    <div class="report-layout">
      <section class="dashboard-card">
        <p class="section-label">Latest Summary</p>
        <h2>最新互動概況</h2>
        <div class="summary-grid">
          <article class="summary-item"><strong>${escapeHtml(String(summary.total_turns || 0))}</strong><span>對話輪數</span></article>
          <article class="summary-item"><strong>${escapeHtml(formatPercent(summary.slot_completion?.用藥狀況?.completion_ratio || 0))}</strong><span>用藥進度</span></article>
          <article class="summary-item"><strong>${escapeHtml(formatPercent(summary.slot_completion?.睡眠狀態?.completion_ratio || 0))}</strong><span>睡眠進度</span></article>
          <article class="summary-item"><strong>${escapeHtml(formatPercent(summary.slot_completion?.作息活動?.completion_ratio || 0))}</strong><span>作息進度</span></article>
          <article class="summary-item"><strong>${escapeHtml(formatPercent(summary.slot_completion?.飲食狀況?.completion_ratio || 0))}</strong><span>飲食進度</span></article>
          <article class="summary-item"><strong>${escapeHtml(summary.latest_emotion?.label || "尚未分析")}</strong><span>最新情緒</span></article>
        </div>
      </section>

      <section class="dashboard-card">
        <p class="section-label">History</p>
        <h2>家族對話紀錄</h2>
        <div class="report-record-list">${renderReportRecordList(report.records || []) || '<p class="panel-text">目前還沒有歷史紀錄。</p>'}</div>
      </section>

      <section class="dashboard-card">
        <p class="section-label">Conversation Timeline</p>
        <h2>最近對話節點</h2>
        <div class="timeline-list">
          ${turns.slice(-8).map((turn) => `
            <article class="timeline-item">
              <span class="timeline-step">Turn ${escapeHtml(turn.turn)}</span>
              <strong>${escapeHtml(turn.target_slot)}</strong>
              <p>${escapeHtml(turn.elder_message)}</p>
            </article>
          `).join("") || '<p class="panel-text">尚未開始對話。</p>'}
        </div>
      </section>

      ${renderPersonaSection(payload.persona_profile || {})}
      ${renderSummarySection(summary, payload)}
    </div>
  `;
}

function renderAdminLoginView() {
  return `
    <section class="auth-layout">
      <article class="auth-card">
        <p class="section-label">Admin Console</p>
        <h2>登入後台</h2>
        <p class="panel-text">從這裡進入後台管理，可調整畫像參數、提示詞、用戶帳號、對話紀錄與 API 設定。</p>
        <form id="admin-login-form" class="auth-form">
          <label class="field">
            <span>後台帳號</span>
            <input id="admin-username" class="text-input" value="admin.console" />
          </label>
          <label class="field">
            <span>後台密碼</span>
            <input id="admin-password" class="text-input" type="password" />
          </label>
          ${state.adminLoginError ? `<p class="form-error">${escapeHtml(state.adminLoginError)}</p>` : ""}
          <button class="send-button" type="submit">登入後台</button>
        </form>
      </article>

      <article class="dashboard-card">
        <p class="section-label">Admin Scope</p>
        <h2>後台可管理項目</h2>
        <div class="chip-list">
          <span class="chip is-muted">畫像參數</span>
          <span class="chip is-muted">提示詞</span>
          <span class="chip is-muted">帳號密碼</span>
          <span class="chip is-muted">對話紀錄</span>
          <span class="chip is-muted">API 設定</span>
        </div>
      </article>
    </section>
  `;
}

function renderUserTableRows(users) {
  return users.map((user) => `
    <tr>
      <td>${escapeHtml(user.display_name)}</td>
      <td><code>${escapeHtml(user.username)}</code></td>
      <td>${escapeHtml(user.role)}</td>
      <td>${escapeHtml(user.persona_label || user.persona_profile_id)}</td>
      <td><code>${escapeHtml(user.password || "")}</code></td>
      <td class="table-actions">
        <button class="ghost-button" data-edit-user-id="${escapeHtml(user.id)}">編輯</button>
        <button class="ghost-button danger" data-delete-user-id="${escapeHtml(user.id)}">刪除</button>
      </td>
    </tr>
  `).join("");
}

function renderAdminView() {
  const adminData = state.adminData || { users: [], personas: {}, prompt_settings: {}, api_settings: {}, conversation_records: [] };
  const personaOptions = Object.keys(adminData.personas || {}).map((profileId) => `
    <option value="${escapeHtml(profileId)}"${profileId === state.adminPersonaId ? " selected" : ""}>${escapeHtml(adminData.personas[profileId].label || profileId)}</option>
  `).join("");
  const personaText = state.adminPersonaId ? JSON.stringify(adminData.personas?.[state.adminPersonaId] || {}, null, 2) : "";
  const userDraft = state.adminUserDraft || getDefaultUserDraft();
  const apiSettings = adminData.api_settings || {};
  const promptSettings = adminData.prompt_settings || {};

  return `
    <section class="hero module-hero">
      <div class="hero-copy">
        <p class="eyebrow">Admin Console</p>
        <h1>平台後台管理</h1>
        <p class="hero-text">這裡可以編輯畫像、調整提示詞、管理帳號密碼、刪修對話紀錄，以及更新 API 與模型預設。</p>
      </div>
      <div class="hero-side">
        <article class="guide-card">
          <span class="highlight-kicker">登入身份</span>
          <strong>${escapeHtml(state.adminUser?.display_name || "管理員")}</strong>
          <p>更新時間：${escapeHtml(adminData.updated_at || "")}</p>
        </article>
      </div>
    </section>

    <div class="admin-grid">
      <section class="dashboard-card">
        <p class="section-label">Persona Manager</p>
        <h2>畫像調整 / 上傳</h2>
        <label class="field">
          <span>選擇畫像</span>
          <select id="admin-persona-select">${personaOptions}</select>
        </label>
        <label class="field">
          <span>畫像 JSON</span>
          <textarea id="admin-persona-json" class="text-area tall">${escapeHtml(personaText)}</textarea>
        </label>
        <div class="inline-actions">
          <button class="send-button" data-action="save-persona">儲存畫像</button>
        </div>
        <label class="field">
          <span>批次上傳 JSON</span>
          <textarea id="admin-persona-import" class="text-area" placeholder='可貼上 {"profile_id": {...}} 的 JSON'></textarea>
        </label>
        <button class="ghost-button" data-action="import-personas">批次匯入</button>
      </section>

      <section class="dashboard-card">
        <p class="section-label">Prompt Manager</p>
        <h2>提示詞調整</h2>
        <label class="field"><span>分析附加提示</span><textarea id="prompt-analysis" class="text-area">${escapeHtml(promptSettings.analysis_appendix || "")}</textarea></label>
        <label class="field"><span>生成附加提示</span><textarea id="prompt-generation" class="text-area">${escapeHtml(promptSettings.generation_appendix || "")}</textarea></label>
        <label class="field"><span>融合模式附加提示</span><textarea id="prompt-fused" class="text-area">${escapeHtml(promptSettings.fused_appendix || "")}</textarea></label>
        <button class="send-button" data-action="save-prompts">儲存提示詞</button>
      </section>

      <section class="dashboard-card">
        <p class="section-label">User Manager</p>
        <h2>帳號密碼與用戶管理</h2>
        <form id="admin-user-form" class="admin-form">
          <input id="user-id" type="hidden" value="${escapeHtml(userDraft.id)}" />
          <label class="field"><span>顯示名稱</span><input id="user-display-name" class="text-input" value="${escapeHtml(userDraft.display_name)}" /></label>
          <label class="field"><span>帳號</span><input id="user-username" class="text-input" value="${escapeHtml(userDraft.username)}" /></label>
          <label class="field"><span>密碼</span><input id="user-password" class="text-input" value="${escapeHtml(userDraft.password)}" /></label>
          <label class="field"><span>角色</span>
            <select id="user-role">
              <option value="family"${userDraft.role === "family" ? " selected" : ""}>family</option>
              <option value="elder"${userDraft.role === "elder" ? " selected" : ""}>elder</option>
              <option value="admin"${userDraft.role === "admin" ? " selected" : ""}>admin</option>
            </select>
          </label>
          <label class="field"><span>家庭畫像</span>
            <select id="user-persona-profile">
              ${Object.keys(adminData.personas || {}).map((profileId) => `<option value="${escapeHtml(profileId)}"${userDraft.persona_profile_id === profileId ? " selected" : ""}>${escapeHtml(adminData.personas[profileId].label || profileId)}</option>`).join("")}
            </select>
          </label>
          <label class="field"><span>預設頁面</span>
            <select id="user-default-view">
              <option value="family"${userDraft.default_view === "family" ? " selected" : ""}>family</option>
              <option value="elder"${userDraft.default_view === "elder" ? " selected" : ""}>elder</option>
              <option value="report"${userDraft.default_view === "report" ? " selected" : ""}>report</option>
              <option value="admin"${userDraft.default_view === "admin" ? " selected" : ""}>admin</option>
            </select>
          </label>
          <label class="toggle-field inline-toggle">
            <span>啟用帳號</span>
            <input id="user-enabled" class="toggle-input" type="checkbox"${userDraft.enabled ? " checked" : ""}>
          </label>
          <div class="inline-actions">
            <button class="send-button" type="submit">儲存帳號</button>
            <button class="ghost-button" type="button" data-action="new-user">新增空白</button>
          </div>
        </form>
        <div class="table-wrap">
          <table class="admin-table">
            <thead>
              <tr><th>名稱</th><th>帳號</th><th>角色</th><th>家庭</th><th>密碼</th><th>操作</th></tr>
            </thead>
            <tbody>${renderUserTableRows(adminData.users || [])}</tbody>
          </table>
        </div>
      </section>

      <section class="dashboard-card">
        <p class="section-label">Conversation Records</p>
        <h2>對話紀錄管理</h2>
        <div class="record-stack">
          ${(adminData.conversation_records || []).map((record) => `
            <article class="record-card">
              <div>
                <strong>${escapeHtml(record.persona_label || record.persona_profile_id || "")}</strong>
                <p>${escapeHtml(record.display_name || record.username || "")}</p>
                <p>${escapeHtml(record.updated_at || "")}</p>
              </div>
              <button class="ghost-button danger" data-delete-record-id="${escapeHtml(record.session_id)}">刪除紀錄</button>
            </article>
          `).join("") || '<p class="panel-text">目前沒有對話紀錄。</p>'}
        </div>
      </section>

      <section class="dashboard-card">
        <p class="section-label">API Manager</p>
        <h2>API 與模型設定</h2>
        <label class="field"><span>預設演算法</span><select id="api-default-algorithm"><option value="dqn"${apiSettings.default_algorithm === "dqn" ? " selected" : ""}>dqn</option><option value="q_learning"${apiSettings.default_algorithm === "q_learning" ? " selected" : ""}>q_learning</option></select></label>
        <label class="field"><span>預設分析模型</span><input id="api-default-analysis" class="text-input" value="${escapeHtml(apiSettings.default_analysis_preset || "")}" /></label>
        <label class="field"><span>預設生成模型</span><input id="api-default-generation" class="text-input" value="${escapeHtml(apiSettings.default_generation_preset || "")}" /></label>
        <label class="field"><span>模型覆寫 JSON</span><textarea id="api-model-overrides" class="text-area tall">${escapeHtml(JSON.stringify(apiSettings.model_overrides || {}, null, 2))}</textarea></label>
        <button class="send-button" data-action="save-api-settings">儲存 API 設定</button>
      </section>
    </div>
  `;
}

function renderView() {
  if (state.currentView === "login") return renderLoginView();
  if (state.currentView === "portal") return renderPortalView();
  if (state.currentView === "family") return renderFamilyView();
  if (state.currentView === "elder") return renderElderView();
  if (state.currentView === "report") return renderReportView();
  if (state.currentView === "admin_login") return renderAdminLoginView();
  if (state.currentView === "admin") return renderAdminView();
  return renderLandingView();
}

function renderApp() {
  root.innerHTML = `
    <div class="app-shell">
      ${renderTopbar()}
      ${renderView()}
    </div>
  `;
  bindEvents();
}

async function handleUserLogin(event) {
  event.preventDefault();
  const username = document.getElementById("login-username").value.trim();
  const password = document.getElementById("login-password").value.trim();
  try {
    const payload = await requestJson("/api/login", { username, password });
    state.authToken = payload.auth_token;
    state.currentUser = payload.user;
    state.currentUser.persona_label = payload.bootstrap?.personas?.[payload.user.persona_profile_id]?.label || state.currentUser.persona_profile_id;
    state.bootstrap = payload.bootstrap;
    state.loginError = "";
    state.sessionId = null;
    state.lastPayload = null;
    state.reportData = null;
    await navigate("portal");
  } catch (error) {
    state.loginError = error.message;
    renderApp();
  }
}

async function handleAdminLogin(event) {
  event.preventDefault();
  const username = document.getElementById("admin-username").value.trim();
  const password = document.getElementById("admin-password").value.trim();
  try {
    const payload = await requestJson("/api/admin/login", { username, password });
    state.adminToken = payload.auth_token;
    state.adminUser = payload.user;
    state.adminLoginError = "";
    await navigate("admin");
  } catch (error) {
    state.adminLoginError = error.message;
    renderApp();
  }
}

async function handleChatSubmit(event) {
  event.preventDefault();
  const textarea = document.getElementById("chat-input");
  const message = textarea.value.trim();
  if (!message) return;
  const payload = await requestJson("/api/chat", {
    auth_token: state.authToken,
    session_id: state.sessionId,
    message,
  });
  textarea.value = "";
  state.lastPayload = payload;
  renderApp();
  if (payload.background_processing) {
    scheduleBackgroundPolling();
  } else {
    stopBackgroundPolling();
  }
}

async function handlePersonaSave() {
  const profileId = document.getElementById("admin-persona-select").value;
  const rawText = document.getElementById("admin-persona-json").value;
  await requestJson("/api/admin/persona/update", {
    auth_token: state.adminToken,
    profile_id: profileId,
    profile: JSON.parse(rawText),
  });
  await loadAdminState();
  renderApp();
}

async function handlePromptSave() {
  await requestJson("/api/admin/prompts/update", {
    auth_token: state.adminToken,
    prompt_settings: {
      analysis_appendix: document.getElementById("prompt-analysis").value,
      generation_appendix: document.getElementById("prompt-generation").value,
      fused_appendix: document.getElementById("prompt-fused").value,
    },
  });
  await loadAdminState();
  renderApp();
}

async function handleUserSave(event) {
  event.preventDefault();
  await requestJson("/api/admin/users/upsert", {
    auth_token: state.adminToken,
    user: {
      id: document.getElementById("user-id").value,
      display_name: document.getElementById("user-display-name").value,
      username: document.getElementById("user-username").value,
      password: document.getElementById("user-password").value,
      role: document.getElementById("user-role").value,
      persona_profile_id: document.getElementById("user-persona-profile").value,
      default_view: document.getElementById("user-default-view").value,
      enabled: document.getElementById("user-enabled").checked,
    },
  });
  state.adminUserDraft = getDefaultUserDraft();
  await loadAdminState();
  renderApp();
}

async function handleApiSave() {
  await requestJson("/api/admin/api/update", {
    auth_token: state.adminToken,
    api_settings: {
      default_algorithm: document.getElementById("api-default-algorithm").value,
      default_analysis_preset: document.getElementById("api-default-analysis").value,
      default_generation_preset: document.getElementById("api-default-generation").value,
      model_overrides: JSON.parse(document.getElementById("api-model-overrides").value),
    },
  });
  state.bootstrap = await requestJson("/api/bootstrap");
  await loadAdminState();
  renderApp();
}

function bindEvents() {
  root.querySelectorAll("[data-action='go-landing']").forEach((button) => button.addEventListener("click", () => {
    state.currentView = "landing";
    renderApp();
  }));
  root.querySelectorAll("[data-action='go-login']").forEach((button) => button.addEventListener("click", () => {
    state.currentView = "login";
    renderApp();
  }));
  root.querySelectorAll("[data-action='go-portal']").forEach((button) => button.addEventListener("click", () => navigate("portal")));
  root.querySelectorAll("[data-action='go-portal-demo']").forEach((button) => button.addEventListener("click", () => {
    state.currentView = state.currentUser ? "portal" : "login";
    renderApp();
  }));
  root.querySelectorAll("[data-action='go-admin']").forEach((button) => button.addEventListener("click", () => navigate(state.adminToken ? "admin" : "admin_login")));
  root.querySelectorAll("[data-action='logout-user']").forEach((button) => button.addEventListener("click", async () => {
    if (state.authToken) {
      await requestJson("/api/logout", { auth_token: state.authToken });
    }
    stopBackgroundPolling();
    state.authToken = "";
    state.currentUser = null;
    state.sessionId = null;
    state.lastPayload = null;
    state.reportData = null;
    state.currentView = "landing";
    renderApp();
  }));
  root.querySelectorAll("[data-fill-username]").forEach((button) => button.addEventListener("click", () => {
    const usernameInput = document.getElementById("login-username");
    const passwordInput = document.getElementById("login-password");
    if (!usernameInput || !passwordInput) return;
    usernameInput.value = button.dataset.fillUsername || "";
    passwordInput.value = button.dataset.fillPassword || "";
  }));
  root.querySelectorAll("[data-action='enter-family']").forEach((button) => button.addEventListener("click", () => navigate("family")));
  root.querySelectorAll("[data-action='enter-elder']").forEach((button) => button.addEventListener("click", () => navigate("elder")));
  root.querySelectorAll("[data-action='enter-report']").forEach((button) => button.addEventListener("click", () => navigate("report")));

  const loginForm = document.getElementById("login-form");
  if (loginForm) loginForm.addEventListener("submit", handleUserLogin);
  const adminLoginForm = document.getElementById("admin-login-form");
  if (adminLoginForm) adminLoginForm.addEventListener("submit", handleAdminLogin);
  const chatForm = document.getElementById("chat-form");
  if (chatForm) chatForm.addEventListener("submit", handleChatSubmit);

  const personaSelect = document.getElementById("admin-persona-select");
  if (personaSelect) {
    personaSelect.addEventListener("change", () => {
      state.adminPersonaId = personaSelect.value;
      renderApp();
    });
  }
  root.querySelectorAll("[data-action='save-persona']").forEach((button) => button.addEventListener("click", async () => handlePersonaSave()));
  root.querySelectorAll("[data-action='import-personas']").forEach((button) => button.addEventListener("click", async () => {
    const rawText = document.getElementById("admin-persona-import").value.trim();
    if (!rawText) return;
    await requestJson("/api/admin/persona/import", { auth_token: state.adminToken, raw_text: rawText });
    await loadAdminState();
    renderApp();
  }));
  root.querySelectorAll("[data-action='save-prompts']").forEach((button) => button.addEventListener("click", async () => handlePromptSave()));
  const adminUserForm = document.getElementById("admin-user-form");
  if (adminUserForm) adminUserForm.addEventListener("submit", handleUserSave);
  root.querySelectorAll("[data-action='new-user']").forEach((button) => button.addEventListener("click", () => {
    state.adminUserDraft = getDefaultUserDraft();
    renderApp();
  }));
  root.querySelectorAll("[data-edit-user-id]").forEach((button) => button.addEventListener("click", () => {
    const target = (state.adminData?.users || []).find((item) => item.id === button.dataset.editUserId);
    if (!target) return;
    state.adminUserDraft = { ...target };
    renderApp();
  }));
  root.querySelectorAll("[data-delete-user-id]").forEach((button) => button.addEventListener("click", async () => {
    await requestJson("/api/admin/users/delete", { auth_token: state.adminToken, user_id: button.dataset.deleteUserId });
    state.adminUserDraft = getDefaultUserDraft();
    await loadAdminState();
    renderApp();
  }));
  root.querySelectorAll("[data-action='save-api-settings']").forEach((button) => button.addEventListener("click", async () => handleApiSave()));
  root.querySelectorAll("[data-delete-record-id]").forEach((button) => button.addEventListener("click", async () => {
    await requestJson("/api/admin/records/delete", { auth_token: state.adminToken, session_id: button.dataset.deleteRecordId });
    await loadAdminState();
    renderApp();
  }));
}

async function init() {
  state.bootstrap = await requestBootstrap();
  state.adminUserDraft = getDefaultUserDraft();
  renderApp();
}

init().catch((error) => {
  root.innerHTML = `
    <div class="app-shell">
      <section class="dashboard-card">
        <h2>初始化失敗</h2>
        <p class="panel-text">${escapeHtml(error.message)}</p>
      </section>
    </div>
  `;
});
