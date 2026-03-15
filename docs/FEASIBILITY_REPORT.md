# 可行性測試報告

測試日期：2026-03-15  
工作目錄：`C:\Users\15507\Desktop\老人`

## 1. 測試目的

本報告的目標是回答三個問題：

1. 這個專案目前是否能被理解與展示
2. 這個專案目前是否能完整重跑
3. 若要上傳 GitHub，哪些風險需要先處理

## 2. 測試方法

本次使用以下方式檢查：

- 檢查主要 Python 檔案語法
- 檢查必要與可選套件是否存在
- 檢查訓練流程依賴的本地模組是否存在
- 檢查 `專案說明/` 是否有實際內容
- 檢查既有 JSON 輸出是否可讀
- 檢查原始碼中是否還殘留 OpenAI 金鑰
- 檢查 GitHub CLI 是否可用

自動化檢查腳本：

```powershell
py tools/feasibility_check.py
```

## 3. 總結結論

本專案目前屬於 `partial feasibility`。

也就是：

- 可做文件補完
- 可做架構重建
- 可做成果網頁展示
- 可做既有輸出資料分析

但尚未達到：

- 完整重現訓練
- 一鍵執行全流程
- 直接部署

## 4. 詳細結果

### 4.1 語法與程式本體

| 檔案 | 結果 |
|---|---|
| `dialogue_simulator.py` | 通過 |
| `integrated_dqn_train.py` | 通過 |
| `R_data.py` | 通過 |
| `script_generator.py` | 通過 |

判讀：

- 主要 Python 檔案本身可通過語法級檢查
- 目前阻礙不在語法，而在依賴與檔案缺漏

### 4.2 套件依賴

#### 已找到

- `numpy`
- `torch`
- `sentence_transformers`
- `pandas`

#### 缺少

- `openai`
- `docx`  
  說明：`docx` 對應常見套件為 `python-docx`

判讀：

- 即使語法可過，仍不能保證直接執行
- 腳本生成與訓練流程至少還缺 3 個套件

### 4.3 本地模組缺漏

目前 RL 訓練核心模組已整理為：

- `dueling_dqn.py`
- `tabular_q_learning.py`
- `integrated_dqn_train.py`

判讀：

- 訓練器已改成預設 DQN、可切換 Q-learning
- 後續阻塞點改成資料完整性與套件安裝，而不是演算法切換結構

### 4.4 參考資料狀態

`專案說明/` 目前是空資料夾。

判讀：

- 無法從使用者提供的參考內容直接補完專案背景
- 本次 README 與報告只能依據現有程式碼與輸出 JSON 重建

### 4.5 既有輸出樣本

已找到並成功解析：

- `grandma_session_20250713_185829/progress.json`
- `pure_dialogue_20250721_142929.json`
- `rl_data_20250721_142929.json`

從樣本推得：

- 腳本數量：12
- 來源類型：3
- 目標槽位：4
- 平均每腳本步數：6.33
- 範例對話輪數：59
- 平均相似度：0.5827
- 平均偏離程度：1.2034
- 轉場腳本使用次數：10

判讀：

- 專案不是空殼
- 先前確實有跑過腳本生成與模擬流程
- 足夠支撐 README 與展示頁面

### 4.6 安全性

原始碼中原本存在硬編碼的 OpenAI API key。

本次已處理：

- `script_generator.py` 改為讀取 `OPENAI_API_KEY`
- `dialogue_simulator.py` 改為讀取 `OPENAI_API_KEY`
- 新增 `.env.example`

最新檢查結果：

- `secrets_found = 0`

判讀：

- 已不適合再把金鑰直接放進程式碼
- 現在的版本已比較適合推上 GitHub

### 4.7 GitHub 上傳能力

本機現況：

- `git` 可用
- `gh`（GitHub CLI）不存在

判讀：

- 可以先做本地 git 初始化與 commit
- 但若沒有 GitHub CLI、token、或既有遠端 URL，無法直接自動建立 GitHub 倉庫並推送

## 5. 風險分級

### 高風險

- 缺少 4 個本地模組，導致訓練流程中斷
- 缺少 `openai` / `jsonlines` / `python-docx`
- `專案說明/` 無內容

### 中風險

- `data/` 與 `outputs/` 來源不完整
- 路徑與樣本輸出之間存在歷史殘留，重現性不足

### 已處理風險

- 硬編碼 API key

## 6. 可做與不可做

### 目前可做

- 閱讀與理解現有架構
- 基於既有 JSON 做結果分析
- 產出 README、報告、網頁
- 做下一階段 AI 輕量化規劃
- 做本地 git 初始化

### 目前不可做

- 無補件情況下重跑 DQN 訓練全流程
- 無補件情況下重跑完整資料前處理與視覺化
- 直接自動上傳到新的 GitHub 倉庫

## 7. 建議修復順序

1. 補齊 `dueling_dqn.py`
2. 補齊 `data_preprocessor.py`
3. 補齊 `visualize_matrices.py`
4. 補齊 `f1_evaluator.py`
5. 安裝 `openai`、`jsonlines`、`python-docx`
6. 補進 `專案說明/` 與原始 `data/`
7. 補 `requirements.txt`
8. 再進行真正的輕量化與效能比較

## 8. 本次新增成果

- 詳盡 `README.md`
- `tools/feasibility_check.py`
- `artifacts/feasibility_report.json`
- 靜態成果頁 `site/index.html`

## 9. 最終判定

這個專案目前最合理的判定是：

> 可展示、可分析、可整理，但尚未達成完整可重現。

換句話說，現在非常適合：

- 先上 GitHub 做研究展示
- 補齊缺件後再往可訓練、可部署版本推進

而不是直接宣稱：

- 已完成可執行產品
- 已具備一鍵訓練能力
