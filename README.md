# AI 輕量化應用於老人對話模擬專案

本專案聚焦在「老人日常對話模擬、腳本生成、獎勵設計、強化學習訓練」，並以此作為 AI 輕量化應用的討論案例。這份 README 不是只解釋原始程式碼在做什麼，也把目前資料夾裡真實存在的檔案、可行性檢查結果、GitHub 上傳前該注意的風險，一次整理完成。

## 1. AI 輕量化是什麼

AI 輕量化（AI Lightweighting）是指：在盡量保留模型效果的前提下，降低模型的運算量、記憶體占用、延遲、部署成本與維運複雜度。它不是單一技術，而是一組工程策略。

常見目標包括：

- 降低 GPU / CPU 推論成本
- 讓模型能在較小裝置或邊緣設備運行
- 減少回應延遲
- 降低 API 呼叫次數與費用
- 簡化系統依賴，提升可部署性
- 增加可維護性與可測試性

## 2. AI 輕量化常見應用

AI 輕量化最常見於下列場景：

- 手機或平板上的語音助理、翻譯、摘要、OCR
- 長照、醫療、客服等需要長時間運作的陪伴型對話系統
- IoT / 邊緣設備上的影像辨識、異常偵測
- 企業內部知識問答，透過小模型或快取減少大模型成本
- 教育、健康追蹤、日常提醒等需要高可用但低成本的 AI 服務

如果把本專案放進這個脈絡，它很適合作為「高齡照護對話系統」的研究原型：先用較完整、較重的流程建立資料與策略，再逐步把核心能力壓縮成更容易部署的版本。

## 3. 這個專案與 AI 輕量化的關聯

這個資料夾中的程式，現在已經可以拆成「離線生成 / 訓練」與「輕量 runtime」兩段：

1. 以背景資訊、喜好興趣、作息等資料生成奶奶對話腳本
2. 模擬奶奶回覆，並根據偏離程度與槽位填充情況記錄 RL 資料
3. 計算 reward，將對話過程轉成 `(state, action, reward, next_state)` 訓練樣本
4. 交給 DQN 或 Q-learning 訓練策略模型
5. 用實際的 RL runtime 根據槽位進度與偏離訊號選下一個劇本，並輸出家屬摘要

其中前 1 到 3 步偏重研究與資料建立，第 4 到 5 步則是目前已能實跑的決策與摘要流程。

這也正是 AI 輕量化最適合發揮的地方：

- 依賴 OpenAI API
- 使用 `SentenceTransformer` 與 `torch`
- 有多輪對話決策成本
- 有摘要與結構化資訊提取需求

因此，這個專案非常適合討論 AI 輕量化，因為它同時具備：

- 對話生成成本
- 向量相似度計算成本
- 強化學習訓練成本
- 多模組、多輸出檔案的工程複雜度

## 4. 專案目前可辨識的架構

依照目前原始碼、輸出檔與新補上的 runtime，專案架構如下：

### 4.1 腳本生成

`script_generator.py`

- 來源維度：`背景資訊`、`喜好興趣`、`作息`
- 目標槽位：`用藥狀況`、`睡眠狀態`、`作息活動`、`飲食狀況`
- 組合結果：`3 × 4 = 12` 組腳本
- 每個腳本包含多輪 child / grandma 對話步驟

### 4.2 對話模擬

`dialogue_simulator.py`

- 用 `SentenceTransformer` 計算語意相似度
- 依相似度轉成偏離程度 `0~3`
- 若偏離程度過高，會切換到轉場腳本
- 同時記錄 RL 訓練所需的 `state/action/reward requirements`

### 4.3 Reward 設計

`R_data.py`

- 即時獎勵由三部分構成：
  - 槽位是否有填到
  - 偏離程度的獎懲
  - 與目標腳本距離是否改善
- 終端獎勵則考慮：
  - 是否完成所有槽位
  - 對話長度是否過長
  - 平均偏離程度是否過高

### 4.4 RL 訓練

`integrated_dqn_train.py`

- 目標是訓練 12 個腳本動作的選擇策略
- 使用 `state_dim=5`、`action_dim=12`
- 現在已改成 **RL 預設走 DQN**，但可以透過參數切換為 **tabular Q-learning**
- 兩種演算法共用同一個主訓練器與相同輸入資料格式

#### 演算法切換方式

```powershell
py integrated_dqn_train.py --algorithm dqn --input rl_data_20250721_142929.json
py integrated_dqn_train.py --algorithm q_learning --input rl_data_20250721_142929.json
```

### 4.5 實際對話 Runtime

`virtual_child_rl_system.py`

- 載入既有 12 組劇本與 RL 訓練資料
- 將對話狀態整理成 `4 個槽位進度 + 1 個偏離旗標`
- 實際讓 DQN / Q-learning 參與「下一個劇本」選擇
- 用 `SentenceTransformer` 偵測偏離，失敗時自動 fallback 到 token overlap
- 用規則式槽位抽取蒐集睡眠、飲食、作息、用藥資訊
- 輸出家屬摘要 Markdown 與完整 transcript JSON

#### Runtime 執行方式

```powershell
py virtual_child_rl_system.py --mode demo --algorithm dqn
py virtual_child_rl_system.py --mode demo --algorithm q_learning
py virtual_child_rl_system.py --mode interactive --algorithm dqn
```

## 5. 已找到且已驗證的輸出成果

根據資料夾中的 JSON 檔，專案目前至少產出過以下結果：

- 12 份奶奶對話腳本
- 平均每份腳本 `6.33` 步
- 最少 `5` 步，最多 `10` 步
- 1 份 59 輪的對話模擬樣本
- 1 份對應的 RL 訓練資料
- 2 份新的 RL runtime 示範成果（DQN / Q-learning 各 1）

### 5.1 歷史樣本對話統計

- 對話輪數：`59`
- 平均相似度：`0.5827`
- 平均偏離程度：`1.2034`
- 使用轉場腳本次數：`10`
- 轉場腳本回合數：`27`
- 成功填到槽位的回合數：`21`

### 5.2 偏離程度分布

- Level 0：`29`
- Level 1：`5`
- Level 2：`9`
- Level 3：`16`

### 5.3 最終槽位填充數量

- `用藥狀況`：6
- `睡眠狀態`：4
- `作息活動`：4
- `飲食狀況`：5

說明：這些數字來自資料夾中既有的 `progress.json`、`pure_dialogue_20250721_142929.json`、`rl_data_20250721_142929.json`，不是假設值。

### 5.4 新增的實跑成果

已新增並驗證：

- `artifacts/runtime_demo/runtime_session_20260315_153557.json`
- `artifacts/runtime_demo/caregiver_summary_20260315_153557.md`
- `artifacts/runtime_demo/runtime_session_20260315_153754.json`
- `artifacts/runtime_demo/caregiver_summary_20260315_153754.md`

這代表專案現在不只是「有資料可展示」，而是能真的跑出：

- RL 決策過的下一句對話
- 偏離後的重聚焦
- 家屬 / 照護端摘要

## 6. 與說明書 / 簡報的對齊程度

我已將 `專案說明/` 內的 `docx` 與 `pptx` 內容抽出並逐項比對，結論是：

- 核心概念已對齊：`虛擬兒女`、`隱性聊天`、`健康槽位`、`偏離偵測`、`RL 選劇本`
- 實作已對齊：DQN 為主、可切 Q-learning、可輸出家屬摘要
- 已補成本地完整 demo：正式互動前端、瀏覽器語音、家屬 dashboard
- 後續仍可擴充：多使用者權限、正式雲端 API、跨期趨勢看板

完整對齊表請看：

- `docs/SPEC_ALIGNMENT.md`

## 7. 為什麼這個專案值得做 AI 輕量化

### 7.1 成本面

目前流程對大型模型與 embedding 都有依賴。如果未來要部署成實際服務，成本會來自：

- 腳本生成時的 API 呼叫
- 對話模擬時的 API 呼叫
- 相似度編碼模型的推論
- 訓練與評估流程的額外資源

但現在已經多了一條更輕的路：

- 劇本可預先生成
- RL runtime 可直接重用既有劇本與訓練資料
- 槽位抽取已可先用規則式完成 base version

### 7.2 延遲面

如果系統要做成陪伴型、即時型長照對話機器人，高延遲會直接傷害使用體驗。

### 7.3 部署面

現況依賴：

- `SentenceTransformer` 與 `torch`
- 新劇本生成仍需要 `openai` 與 `python-docx`
- 若要完整重建原始資料流程，仍需要 `data/` 來源檔

但相較前一版，專案已不再是只有分析與文件化，而是能夠執行輕量 runtime。

## 8. 建議的 AI 輕量化路線

以下是最適合這個專案的輕量化順序。

### 8.1 第一階段：先把系統變乾淨

- 移除硬編碼金鑰，改用環境變數
- 補齊 `README`、測試腳本與可行性報告
- 補上依賴清單與資料說明

這一階段不是「模型變小」，但它是後續一切輕量化的前提。

### 8.2 第二階段：推論與資料流程輕量化

- 把 LLM 產生的腳本改成「預生成 + 快取」
- 對相似度模型做批次化、快取或換成更小模型
- 把規則型判斷優先化，減少每輪都呼叫大模型
- 將槽位填充抽取改為規則 + 小模型混合式設計

### 8.3 第三階段：模型壓縮

- Quantization：把可量化模型壓到 8-bit / 4-bit
- Distillation：把大模型知識蒸餾到小模型
- PEFT / LoRA：保留能力但降低訓練成本
- ONNX / TensorRT / llama.cpp：針對推論做部署優化

### 8.4 第四階段：產品化導向

- 將「腳本生成」與「即時對話」拆成離線 / 線上兩段
- 對陪伴型功能保留高品質模型
- 對結構化任務（提醒、追問、分類、槽位填寫）改用小模型或規則
- 對照護現場設備限制，設計雲端版與邊緣版兩種架構

## 9. 可行性測試摘要

我已經補上一個檢查腳本：

```powershell
py tools/feasibility_check.py
```

最新檢查結果：

| 項目 | 結果 | 說明 |
|---|---|---|
| Python 語法檢查 | 通過 | 5 支主要 Python 檔皆可做語法級檢查 |
| 必要套件 | 通過 | `numpy`、`torch`、`sentence_transformers`、`pandas` 可被發現 |
| 可選套件 | 通過 | `.venv` 環境內已補上 `openai`、`python-docx` |
| 本地模組 | 通過 | 已補上 `dueling_dqn.py`、`tabular_q_learning.py`、互動前端與實跑 runtime |
| 參考資料夾 | 存在 | `專案說明/` 已有 `docx` 與 `pptx` 參考檔 |
| 範例輸出 | 存在 | 已找到腳本、對話、RL 樣本、Web 前端與 runtime demo |
| 金鑰外洩 | 已修正 | 已改為從 `OPENAI_API_KEY` 讀取 |

### 目前結論

本專案在 `.venv` 環境下已屬於「可重現（ready）」：

- 文件整理、架構重建、成果展示：可行
- DQN / Q-learning runtime 對話示範：可行
- 家屬摘要輸出：可行
- 正式互動前端、瀏覽器語音與家屬 dashboard：可行
- 重新生成新劇本與完整雲端部署：可進一步擴充

目前仍值得持續補強的部分是：

- 多使用者與正式權限管理
- 雲端 API 化
- 長期照護趨勢 dashboard

## 10. GitHub 上傳前檢查清單

- [x] 移除硬編碼 OpenAI API key
- [x] 新增 `.env.example`
- [x] 新增 `.gitignore`
- [x] 新增自動化可行性檢查腳本
- [x] 整理詳細 README
- [x] 產出可行性報告
- [x] 補上說明書 / 簡報對齊表
- [x] 建立可實跑的 RL runtime
- [x] 製作靜態成果網頁
- [ ] 建立 GitHub 遠端倉庫
- [ ] 推送到 GitHub

最後兩步是否能完成，取決於本機是否已具備 GitHub CLI 或可用的 GitHub 驗證資訊。

## 11. 建議的下一步

如果你要把這個專案繼續往研究或產品化推進，最建議的順序是：

1. 把 `care_companion_server.py` 擴成多使用者正式 API
2. 把目前單次摘要升級成跨日 / 跨週照護趨勢板
3. 補齊更多個案資料與測試語料
4. 規劃 Hugging Face / 雲端部署版
5. 再做真正的 AI 輕量化驗證，如量化、蒸餾、推論延遲比較

## 12. 這次新增的成果

- `README.md`
- `docs/FEASIBILITY_REPORT.md`
- `docs/SPEC_ALIGNMENT.md`
- `tools/feasibility_check.py`
- `artifacts/feasibility_report.json`
- `site/index.html`
- `site/styles.css`
- `site/app.js`
- `.env.example`
- `virtual_child_rl_system.py`
- `care_companion_server.py`
- `care_frontend/`

## 13. 快速啟動

### 13.1 建立與啟用虛擬環境

如果還沒建立虛擬環境，可在專案根目錄執行：

```powershell
py -3.13 -m venv .venv
```

啟用方式：

```powershell
.\.venv\Scripts\Activate.ps1
```

安裝依賴：

```powershell
pip install -r requirements.txt
```

### 13.2 設定 API 金鑰（只有生成新劇本時需要）

PowerShell：

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

### 13.3 執行可行性檢查

```powershell
.\.venv\Scripts\python.exe tools\feasibility_check.py
```

### 13.4 執行 RL runtime

```powershell
.\.venv\Scripts\python.exe virtual_child_rl_system.py --mode demo --algorithm dqn
.\.venv\Scripts\python.exe virtual_child_rl_system.py --mode demo --algorithm q_learning
.\.venv\Scripts\python.exe virtual_child_rl_system.py --mode interactive --algorithm dqn
```

### 13.5 啟動正式互動前端

```powershell
.\.venv\Scripts\python.exe care_companion_server.py --open-browser
```

啟動後瀏覽器會開啟：

```text
http://127.0.0.1:8000
```

### 13.6 開啟成果網頁

直接用瀏覽器打開：

```text
site/index.html
```

---

如果把這份 README 濃縮成一句話，這個專案目前最適合被定位成：

> 一個以高齡對話模擬與強化學習為主題、已能實跑 DQN / Q-learning、正式互動前端、瀏覽器語音與家屬 dashboard 的研究型原型；目前已具備完整本地 demo，下一步是多使用者與雲端化。
