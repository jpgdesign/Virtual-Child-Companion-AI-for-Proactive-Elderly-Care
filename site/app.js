const projectData = {
  readiness: "partial",
  summaryCards: [
    { value: "12", label: "已找到腳本", detail: "3 個來源 × 4 個槽位" },
    { value: "59", label: "樣本對話輪數", detail: "附有對應 RL 資料" },
    { value: "0.5827", label: "平均相似度", detail: "示意對話保真程度" },
    { value: "10", label: "轉場腳本使用次數", detail: "高偏離時會切換" }
  ],
  statusBullets: [
    "主要 Python 檔可通過語法檢查。",
    "README、可行性報告與成果網頁已補齊。",
    "OpenAI 金鑰已改為環境變數讀取。",
    "RL 已改成預設 DQN，並可切換到 tabular Q-learning。",
    "目前主要缺口已縮小為資料完整性與 openai / python-docx 套件。"
  ],
  architecture: [
    {
      step: "01",
      title: "腳本生成",
      file: "script_generator.py",
      body: "從背景資訊、喜好興趣與作息三類資料，生成 12 組奶奶對話腳本，對應 4 個目標槽位。"
    },
    {
      step: "02",
      title: "對話模擬",
      file: "dialogue_simulator.py",
      body: "用語意相似度判斷回覆是否偏離目標腳本，必要時切入轉場腳本，並同步蒐集 RL 狀態。"
    },
    {
      step: "03",
      title: "Reward 計算",
      file: "R_data.py",
      body: "把槽位填充、偏離程度與距離改善轉為即時獎勵，再加上終端獎勵形成完整訓練樣本。"
    },
    {
      step: "04",
      title: "DQN 訓練",
      file: "integrated_dqn_train.py",
      body: "目標是從 12 個腳本動作中學出較佳選擇策略，但目前仍受缺失模組與套件影響。"
    }
  ],
  artifactMetrics: [
    { value: "6.33", label: "平均每份腳本步數", detail: "最少 5 步，最多 10 步" },
    { value: "21", label: "槽位成功填充回合", detail: "代表樣本對話中確實發生資訊補齊" },
    { value: "27", label: "轉場腳本回合", detail: "顯示系統有處理高偏離情境的設計" },
    { value: "4", label: "目標槽位類型", detail: "用藥、睡眠、作息活動、飲食" }
  ],
  artifactFiles: [
    "grandma_session_20250713_185829/progress.json",
    "pure_dialogue_20250721_142929.json",
    "rl_data_20250721_142929.json",
    "artifacts/feasibility_report.json"
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
      body: "4 支主要檔案都能通過語法級檢查，代表程式結構仍可解析。"
    },
    {
      status: "pass",
      title: "必要套件",
      body: "已找到 numpy、torch、sentence_transformers、pandas。"
    },
    {
      status: "warn",
      title: "可選套件",
      body: "目前只剩 openai 與 python-docx 需要依實際執行需求安裝。"
    },
    {
      status: "pass",
      title: "RL 切換",
      body: "已補上 dueling_dqn.py 與 tabular_q_learning.py，主訓練器可在 DQN / Q-learning 間切換。"
    },
    {
      status: "warn",
      title: "參考資料夾",
      body: "專案說明目前是空的，文件重建只能依現有程式與輸出檔進行。"
    },
    {
      status: "pass",
      title: "金鑰安全",
      body: "硬編碼 OpenAI key 已移除，改為讀取 OPENAI_API_KEY。"
    }
  ],
  roadmap: [
    {
      title: "清理與補件",
      body: "先把敏感資訊、文件、套件與模組依賴整理乾淨，讓專案具備可維護性。"
    },
    {
      title: "流程輕量化",
      body: "將腳本生成改為預生成或快取，減少每次都呼叫大模型的成本。"
    },
    {
      title: "推論優化",
      body: "對 embedding 與槽位抽取採取快取、小模型替換與規則優先的混合策略。"
    },
    {
      title: "模型壓縮",
      body: "再進一步做 quantization、distillation、LoRA 或 ONNX 化，走向真正可部署版本。"
    }
  ]
};

const readinessText = {
  ready: "已可重現",
  partial: "部分可行",
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
renderArtifactMetrics();
renderArtifactFiles();
renderDeviationChart();
renderAudit();
renderRoadmap();
