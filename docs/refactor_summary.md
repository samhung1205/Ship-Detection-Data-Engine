# Refactor Summary

這份文件是目前這一輪重構的交接摘要，目標是讓後續回頭看專案時，可以快速知道：

- 這次到底改了哪些主線
- 哪些風險已經被處理
- 目前專案停在哪個合理位置
- 後續什麼情況下才需要再繼續重構

## 1. 這一輪重構的總目標

這次不是重寫專案，而是把原本高度集中在 `gui/main_window.py` 的高風險責任，逐步拆成：

- `sdde/` 持有資料與規則
- `gui/` 的 controllers 處理互動與轉接
- `main_window.py` 主要負責 UI 接線與 orchestration

重點是先處理最容易出錯的平行 state 與重複邏輯，而不是追求完全形式上的分層。

## 2. 已完成的主線整理

### A. 驗證基線與文件欄杆

- 新增 `docs/manual_smoke_test.md`
- 新增 `docs/state_refactor_inventory.md`
- 新增 `docs/paste_refactor_inventory.md`
- README 已同步反映重構後的主線與文件入口

目的：

- 先有手動回歸清單
- 先把 state 與拆分風險盤點清楚
- 避免後續修改只「看起來可用」，但內部 state 已經偏掉

### B. 類別定義與 Label IO 單一路徑化

- 主流程已統一為 `ClassMappingDialog + classes.yaml`
- `gui/dialogs/input_window.py` 與 `data.yaml` 已降為 legacy compatibility
- GUI 的 label 匯入匯出已收斂到 `sdde/import_export.py`
- 補上 `sdde/legacy_rows.py` 作為 legacy GUI row 與資料模型間的轉接
- Bounding Boxes 輸出格式已統一為 `class_id x1 y1 x2 y2`

目的：

- 減少 GUI 和 service layer 各自維護一份格式邏輯
- 降低新舊類別定義混用的風險

### C. GT 主線重構

GT 標註主線目前已整理成：

- `sdde/document.py` 的 `AnnotationDocument`
- `gui/annotation_controller.py`
- `gui/annotation_actions_controller.py`
- `gui/annotation_draw_controller.py`
- `gui/annotation_list_view.py`
- `gui/annotation_list_controller.py`
- `gui/annotation_preview_controller.py`
- `gui/annotation_workspace_controller.py`

已處理的責任包括：

- GT row truth state
- add / rename / delete / clear
- 手動畫框互動
- 列表投影與右鍵選單
- 高亮預覽
- 屬性面板同步
- autosave recovery 後的 GT 對齊
- metadata export / statistics / error analysis 取值

目的：

- 把 `data / real_data / box_attributes` 從主視窗抽成單一 truth source
- 降低平行 list 不同步的風險

### D. Paste 主線重構

Paste 主線目前已整理成：

- `sdde/paste_document.py` 的 `PasteDocument`
- `sdde/paste_candidate.py` 的 `PasteCandidateSession`
- `gui/paste_actions_controller.py`
- `gui/paste_preview_controller.py`
- `gui/paste_candidate_controller.py`

已處理的責任包括：

- committed paste state
- candidate / placement state
- Add / Rename / Delete / Delete all
- 列表高亮預覽
- asset 載入
- horizontal flip
- placement click
- candidate overlay 計算

目的：

- 把 paste 的 committed state 與 candidate state 拆開
- 降低 `pimg_data / real_pimg_data / paste_images / paste_records` 的同步風險
- 避免 `main_window.py` 直接持有過多暫存欄位

## 3. 這一輪順手修掉的重要問題

- 修掉底部狀態列與右下 slider 被遮擋的 GUI 版面問題
- 修掉 paste `Add` 時因 class name 被當成整數轉換而崩潰的 bug
- 修掉 paste `Add` 提交時因 `PasteDocument` 嘗試 `deepcopy(QImage)` 而崩潰的 bug
- 修掉 `recalc_size_tag()` 會被舊值覆蓋的問題

## 4. 測試與回歸狀態

- 測試數量從最初的 `54` 個提升到目前的 `128` 個
- 已補上的重點保護：
  - GT document API
  - GT action / draw / list / preview / workspace controllers
  - paste document API
  - paste candidate session
  - paste action / preview / candidate controllers
  - label import/export adapter
  - legacy paste bbox row 解析

目前完整回歸基線：

- `128 passed`

## 5. 目前專案停在哪裡

目前我認為這個專案已到一個合理的結構停點：

- 高風險的 GT 主線已有清楚 owner
- 高風險的 paste 主線也已有清楚 owner
- `main_window.py` 雖然仍存在，但主要已是接線與流程協調
- README、盤點文件、測試欄杆都有跟上

這代表後續不需要再為了「更乾淨一點」而持續拆分。

## 6. 目前估計完成度

如果以「主要高風險 state 是否有 owner、核心主線是否已收斂、測試與文件是否齊備」來看：

- 目前整體重構完成度約為 `85%`

剩下的 15% 多半屬於：

- 更細的 OpenCV / Qt 細節抽離
- 更完整的 GUI 整合測試
- 視需要再做的局部優化

這些都已經進入報酬遞減區，沒有明確 bug 或需求時不建議主動再做。

## 7. 後續建議

接下來最合理的節奏是：

1. 先以實機操作為主
2. 有 bug 再做精準修補
3. 有新功能需求再做對應擴充
4. 沒有明確價值時，不再做結構性重構

## 8. 一句話結論

這一輪重構已經把專案從「主視窗承擔過多責任、平行 state 風險高」整理到「GT 與 paste 主線都有清楚 owner、測試與文件同步跟上」的狀態，後續應以實際使用回饋驅動修改，而不是繼續為拆而拆。
