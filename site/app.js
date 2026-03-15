const projectData = {
  readiness: "partial",
  summaryCards: [
    { value: "12", label: "劇本總數", detail: "3 個來源 × 4 個槽位" },
    { value: "2", label: "已驗證 Runtime", detail: "DQN / Q-learning 各 1 次" },
    { value: "6", label: "DQN Demo 輪數", detail: "已輸出家屬摘要" },
    { value: "5", label: "核心模組", detail: "從腳本到 runtime 全部打通" }
  ],
  statusBullets: [
    "已對照說明書與簡報，核心主軸改為真正可跑的虛擬兒女 RL 原型。",
    "RL 現在不只負責訓練，還會在 runtime 實際選擇下一個劇本。",
    "DQN 是預設演算法，但可切換到 tabular Q-learning。",
    "系統已能輸出家屬 / 照護摘要，不再只有對話資料。",
    "目前最大缺口已轉為產品化前端、語音化與照護後台。"
  ],
  architecture: [
    {
      step: "01",
      title: "腳本生成",
      file: "script_generator.py",
      body: "從背景資訊、喜好興趣與作息三類資料，生成 12 組奶奶對話腳本。"
    },
    {
      step: "02",
      title: "對話模擬",
      file: "dialogue_simulator.py",
      body: "計算相似度、判斷偏離程度，並在高偏離時切入轉場劇本。"
    },
    {
      step: "03",
      title: "Reward 計算",
      file: "R_data.py",
      body: "把槽位填充、偏離程度與距離改善轉成訓練用 reward。"
    },
    {
      step: "04",
      title: "RL 訓練",
      file: "integrated_dqn_train.py",
      body: "主訓練器可在 DQN 與 Q-learning 間切換，持續使用同一份 RL 資料。"
    },
    {
      step: "05",
      title: "實跑 Runtime",
      file: "virtual_child_rl_system.py",
      body: "RL 會根據槽位進度與偏離訊號選下一個劇本，並輸出家屬摘要。"
    }
  ],
  specAlignment: [
    {
      status: "pass",
      title: "虛擬兒女人設",
      body: "系統已用孫女 / 奶奶劇本與陪伴式語氣實際落地。"
    },
    {
      status: "pass",
      title: "偏離重聚焦",
      body: "對話偏離後會切換重聚焦話術，不再只是單純續聊。"
    },
    {
      status: "pass",
      title: "RL 選劇本",
      body: "DQN / Q-learning 會實際參與下一個腳本選擇。"
    },
    {
      status: "pass",
      title: "家屬摘要",
      body: "已可輸出 Markdown 摘要與 JSON transcript。"
    },
    {
      status: "warn",
      title: "正式前端",
      body: "目前仍以成果頁與 CLI runtime 為主，尚未做互動式照護前端。"
    },
    {
      status: "warn",
      title: "語音能力",
      body: "STT / TTS 仍未接入，現在先完成文字版核心流程。"
    }
  ],
  artifactMetrics: [
    { value: "0.516", label: "DQN Demo 平均相似度", detail: "6 輪對話後仍可持續追問" },
    { value: "3", label: "DQN Demo 轉場次數", detail: "代表偏離時有真的重聚焦" },
    { value: "100%", label: "睡眠槽完成度", detail: "DQN demo 已完整蒐集睡眠資訊" },
    { value: "75%", label: "用藥槽完成度", detail: "代表摘要已能反映蒐集缺口" }
  ],
  artifactFiles: [
    "docs/SPEC_ALIGNMENT.md",
    "artifacts/runtime_demo/runtime_session_20260315_153557.json",
    "artifacts/runtime_demo/caregiver_summary_20260315_153557.md",
    "artifacts/runtime_demo/runtime_session_20260315_153754.json",
    "artifacts/runtime_demo/caregiver_summary_20260315_153754.md"
  ],
  deviationCounts: [
    { label: "Level 0", value: 29 },
    { label: "Level 1", value: 5 },
    { label: "Level 2", value: 9 },
    { label: "Level 3", value: 16 }
  ],
  audit: [
    {
      status: "pass",
      title: "Python 語法",
      body: "5 支主要 Python 檔都可通過語法檢查。"
    },
    {
      status: "pass",
      title: "RL Runtime",
      body: "DQN 與 Q-learning 都已實跑並產出對話與摘要。"
    },
    {
      status: "pass",
      title: "參考資料",
      body: "專案說明資料夾已有 docx / pptx，對齊工作已可直接比對。"
    },
    {
      status: "warn",
      title: "可選套件",
      body: "若要重新生成新劇本，仍需 openai 與 python-docx。"
    },
    {
      status: "warn",
      title: "產品化缺口",
      body: "目前尚未補上互動前端、語音 I/O 與照護 dashboard。"
    },
    {
      status: "pass",
      title: "金鑰安全",
      body: "硬編碼 OpenAI key 已移除，改為讀取 OPENAI_API_KEY。"
    }
  ],
  roadmap: [
    {
      title: "接前端",
      body: "把目前 CLI runtime 包成長者可用的 Web / 行動端互動介面。"
    },
    {
      title: "補語音",
      body: "加入 STT / TTS，讓陪伴型對話更接近實際使用場景。"
    },
    {
      title: "做儀表板",
      body: "把家屬摘要升級為趨勢 dashboard 與異常提示。"
    },
    {
      title: "做輕量化驗證",
      body: "針對 embedding、策略模型與摘要流程做延遲與成本比較。"
    }
  ]
};

const readinessText = {
  ready: "已可重現",
  partial: "核心可跑",
  blocked: "目前阻塞"
};

const statusText = {
  pass: "Pass",
  warn: "Warn",
  fail: "Fail"
};

function renderSummaryCards() {
  const root = document.getElementById("summary-cards");
  projectData.summaryCards.forEach((card) => {
    const article = document.createElement("article");
    article.className = "summary-card";
    article.innerHTML = `
      <strong>${card.value}</strong>
      <div>${card.label}</div>
      <span>${card.detail}</span>
    `;
    root.appendChild(article);
  });
}

function renderStatusBullets() {
  const root = document.getElementById("status-bullets");
  projectData.statusBullets.forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    root.appendChild(item);
  });
}

function renderArchitecture() {
  const root = document.getElementById("architecture-grid");
  projectData.architecture.forEach((item) => {
    const card = document.createElement("article");
    card.className = "architecture-card";
    card.dataset.step = item.step;
    card.innerHTML = `
      <p class="section-label">Step ${item.step}</p>
      <h3>${item.title}</h3>
      <code>${item.file}</code>
      <p>${item.body}</p>
    `;
    root.appendChild(card);
  });
}

function renderSpecAlignment() {
  const root = document.getElementById("spec-grid");
  projectData.specAlignment.forEach((item) => {
    const card = document.createElement("article");
    card.className = "audit-item";
    card.innerHTML = `
      <div class="audit-status status-${item.status}">${statusText[item.status]}</div>
      <h3>${item.title}</h3>
      <p>${item.body}</p>
    `;
    root.appendChild(card);
  });
}

function renderArtifactMetrics() {
  const root = document.getElementById("artifact-metrics");
  projectData.artifactMetrics.forEach((item) => {
    const block = document.createElement("div");
    block.className = "metric-item";
    block.innerHTML = `
      <strong>${item.value}</strong>
      <div>${item.label}</div>
      <p>${item.detail}</p>
    `;
    root.appendChild(block);
  });
}

function renderArtifactFiles() {
  const root = document.getElementById("artifact-files");
  projectData.artifactFiles.forEach((file) => {
    const pill = document.createElement("span");
    pill.className = "file-pill";
    pill.textContent = file;
    root.appendChild(pill);
  });
}

function renderDeviationChart() {
  const root = document.getElementById("deviation-chart");
  const max = Math.max(...projectData.deviationCounts.map((item) => item.value));

  projectData.deviationCounts.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-meta">
        <span>${item.label}</span>
        <span>${item.value}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${(item.value / max) * 100}%"></div>
      </div>
    `;
    root.appendChild(row);
  });
}

function renderAudit() {
  const root = document.getElementById("audit-grid");
  projectData.audit.forEach((item) => {
    const card = document.createElement("article");
    card.className = "audit-item";
    card.innerHTML = `
      <div class="audit-status status-${item.status}">${statusText[item.status]}</div>
      <h3>${item.title}</h3>
      <p>${item.body}</p>
    `;
    root.appendChild(card);
  });
}

function renderRoadmap() {
  const root = document.getElementById("roadmap");
  projectData.roadmap.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "roadmap-card";
    card.innerHTML = `
      <div class="roadmap-index">${index + 1}</div>
      <h3>${item.title}</h3>
      <p>${item.body}</p>
    `;
    root.appendChild(card);
  });
}

function renderReadiness() {
  const chip = document.getElementById("readiness-chip");
  chip.textContent = readinessText[projectData.readiness] || projectData.readiness;
}

renderReadiness();
renderSummaryCards();
renderStatusBullets();
renderArchitecture();
renderSpecAlignment();
renderArtifactMetrics();
renderArtifactFiles();
renderDeviationChart();
renderAudit();
renderRoadmap();
