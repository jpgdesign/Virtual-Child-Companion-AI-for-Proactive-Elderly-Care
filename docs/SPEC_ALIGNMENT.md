# 規格對齊表

比對來源：

- `專案說明/附件三：作品說明書(1).docx`
- `專案說明/無水印.pptx`

比對日期：2026-03-15

## 目前結論

目前專案已經和說明書、簡報的核心主軸對齊到「完整本地 demo 可實跑」的程度。

已實作完成的核心：

- `虛擬兒女` 陪伴式對話脈絡
- `3 個話題來源 × 4 個目標槽位 = 12 個劇本`
- 語意偏離偵測
- 高偏離時的重聚焦轉場
- 健康槽位蒐集
- `DQN` 預設、可切換 `Q-learning`
- 家屬 / 照護摘要輸出
- 正式互動前端
- 瀏覽器語音輸入 / 輸出
- 本地照護 dashboard

仍屬後續擴充的項目：

- 多使用者後端 API、權限、日誌
- 跨日 / 跨週趨勢儀表板
- 穿戴裝置與醫療系統整合

## 條目比對

| 說明書 / 簡報需求 | 目前狀態 | 對應實作 |
|---|---|---|
| 虛擬兒女人設陪伴長者 | 已完成 | `script_generator.py` 的孫女 / 奶奶劇本設計，`virtual_child_rl_system.py` 的實際對話流程 |
| 以隱性聊天蒐集健康訊號 | 已完成 | 12 組劇本都從生活話題引導到睡眠、飲食、作息、用藥 |
| 話題來源為背景資訊、喜好興趣、作息 | 已完成 | `script_generator.py` 使用這 3 類來源 |
| 目標槽為用藥、睡眠、作息活動、飲食 | 已完成 | 劇本、模擬器、runtime、摘要皆沿用這 4 大槽位 |
| 語意偏離偵測與重聚焦 | 已完成 | `dialogue_simulator.py`、`virtual_child_rl_system.py` 皆有偏離分級與轉場機制 |
| 用 embedding 做相似度 | 已完成 | `dialogue_simulator.py` 與 `virtual_child_rl_system.py` 使用 `SentenceTransformer`，失敗時有 fallback |
| RL 決定換話題 / 選劇本 | 已完成 | `virtual_child_rl_system.py` 會實際載入或訓練 DQN / Q-learning 並據此選腳本 |
| DQN 為主，可切換 Q-learning | 已完成 | `integrated_dqn_train.py`、`virtual_child_rl_system.py` 都支援切換 |
| 產出家屬 / 照護端摘要 | 已完成 | `virtual_child_rl_system.py` 可輸出 Markdown 摘要與 JSON transcript |
| 系統要真的能跑 | 已完成 | `py virtual_child_rl_system.py --mode demo --algorithm dqn` 與 `--algorithm q_learning` 已實測通過 |
| Web / 行動裝置介面 | 已完成 | `care_companion_server.py` + `care_frontend/` 已提供正式互動式照護前端 |
| 語音輸入 / 輸出 | 已完成 | 前端已接上瀏覽器 STT / TTS |
| 照護端儀表板 | 已完成 | 前端右側 dashboard 已即時顯示槽位進度、提醒與摘要 |
| 權限、日誌、後端 API | 部分完成 | 已有本地 API server，正式多使用者權限仍待擴充 |

## 實際可跑指令

```powershell
py virtual_child_rl_system.py --mode demo --algorithm dqn
py virtual_child_rl_system.py --mode demo --algorithm q_learning
py virtual_child_rl_system.py --mode interactive --algorithm dqn
.\.venv\Scripts\python.exe care_companion_server.py --open-browser
```

## 已驗證輸出

- `artifacts/runtime_demo/runtime_session_20260315_153557.json`
- `artifacts/runtime_demo/caregiver_summary_20260315_153557.md`
- `artifacts/runtime_demo/runtime_session_20260315_153754.json`
- `artifacts/runtime_demo/caregiver_summary_20260315_153754.md`

## 你現在可以怎麼看這個專案

最準確的說法是：

> 這個專案已經不是只有文件或簡報概念，而是有一個可實跑的 RL 陪伴對話本地完整 demo；接下來重點會放在多使用者、雲端 API 與長期趨勢化。
