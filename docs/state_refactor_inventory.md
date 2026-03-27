# Main Window State Inventory

這份文件是 `gui/main_window.py` 拆分前的狀態盤點與保護清單。目標不是一次重寫，而是在每一步重構前先鎖住目前真正需要被同步的資料，避免 UI 看起來正常、內部 state 卻已經偏掉。若接下來要整理 copy-paste augmentation，請改看 `docs/paste_refactor_inventory.md`。

目前第一步已落地：GT 狀態已先集中到 `sdde/document.py` 的 `AnnotationDocument`，而且 `AnnotationController`、屬性面板同步、autosave recovery、metadata export 都已開始改成呼叫它的 API。GT 的 add / rename / delete / clear 入口已集中到 `gui/annotation_actions_controller.py`，手動畫框互動已集中到 `gui/annotation_draw_controller.py`，列表與計數的 UI 投影已集中到 `gui/annotation_list_view.py`，列表選取與右鍵互動則集中到 `gui/annotation_list_controller.py`，高亮預覽已集中到 `gui/annotation_preview_controller.py`，選取後的屬性面板同步則集中到 `gui/annotation_workspace_controller.py`。Paste 的 committed state 現在也已先集中到 `sdde/paste_document.py`，candidate / placement state 已先集中到 `sdde/paste_candidate.py`，已提交動作入口、高亮預覽、以及 asset / transform / placement 流程則集中到 `gui/paste_actions_controller.py`、`gui/paste_preview_controller.py`、`gui/paste_candidate_controller.py`。GUI 表面介面仍維持舊型態，以降低行為變更風險。

## 1. 目前主視窗持有的核心狀態

### A. 類別對照

- `class_catalog`
- `object_list`

來源：

- 啟動時由 `classes.yaml` 載入
- `ClassMappingDialog` 儲存後回寫
- 手動新增框、prediction accept、paste rename 也可能把新類別 append 進 `object_list`

### B. 手動畫框 GT 狀態

- `data`
- `real_data`
- `box_attributes`
- `listwidget`
- `label_list`

用途：

- `data` 是當前畫布尺寸下的顯示座標列
- `real_data` 是原圖像素座標列
- `box_attributes` 以 index 對齊每一筆 `real_data`
- `listwidget` / `label_list` 是同一份 GT 狀態的 UI 投影

### C. Paste augmentation 狀態

- `pimg_data`
- `real_pimg_data`
- `paste_images`
- `paste_records`
- `pimglistwidget`
- `pimg_list`

用途：

- `pimg_data` / `real_pimg_data` 與手動畫框類似，但只屬於 paste 產生的框
- `paste_images` 保存實際貼圖 overlay 的顯示資料
- `paste_records` 保存匯出的 metadata
- committed paste truth state 現在由 `PasteDocument` 持有，主視窗保留 legacy 相容 property

### D. Prediction 狀態

- `predictions`
- `pred_listwidget`

用途：

- 模型預測疊圖
- `Accept` 會把 prediction 轉進 GT 狀態
- `Reject` / `Clear predictions` 只影響 prediction 狀態

### E. 影像 / 畫布 / 視圖狀態

- `imgfilePath`
- `origin_canvas`
- `origin_width`
- `origin_height`
- `ratio_value`
- `hideBox`
- `_tile_grid`
- tile panel 當前設定與 index

用途：

- 這些是顯示層 state，不應成為 annotation truth source

### F. Paste 編輯中的暫存狀態

- `_current_asset_path`
- `pasteimg`
- `origin_pasteimg`
- `resizeimg`
- `rotated`
- `bc_image`
- `norm_pimg`
- `bbox_pimg`
- `real_bbox_pimg`
- `cX` / `cY`

用途：

- 只在 paste 流程中短暫存在
- 不應與已提交的 `pimg_data` / `paste_records` 混淆
- 目前這些 state 已由 `PasteCandidateSession` 持有，主視窗只保留使用它的流程方法

### G. 持久化 / 系統流程狀態

- `_project_config`
- `_autosave_timer`
- `_annotation_controller`

## 2. 目前最重要的不變條件

### GT 手動畫框

以下幾組資料目前是靠 index 對齊的，拆分前一定要一起維護：

- `len(data) == len(real_data) == listwidget.count() == len(box_attributes)`
- `label_list` 顯示的數量要和 `len(real_data)` 一致
- `data[i][0]`、`real_data[i][0]`、`listwidget.item(i).text()` 應該是同一個 class name

### Paste augmentation

目前 paste 也是多份平行狀態：

- `len(pimg_data) == len(real_pimg_data) == len(paste_images) == pimglistwidget.count()`
- 在同一個 session 內新增的 paste，`paste_records` 應與 `real_pimg_data` 維持相同順序

### 類別對照

- `real_data`、`real_pimg_data`、`predictions` 中出現的 class name，都應該可在 `object_list` 找到
- `object_list` 是 GUI legacy 流程仍在使用的查表來源，但權威定義已改為 `class_catalog`

### 影像生命週期

- `imgfilePath`、`origin_canvas`、`origin_width`、`origin_height` 應視為同一組影像 session state
- `newFile()` 重置時，GT / paste / prediction / attribute / selection 都要一起清空

### Autosave

- autosave 只保存 `real_data + box_attributes + object_list`
- recovery 後 `box_attributes` 必須和還原出的 GT index 對齊

## 3. 高風險同步點

這些方法是後續拆 `Document/Session` 時最容易出現資料不同步的地方：

1. `AnnotationDrawController.handle_press()` + `AnnotationActionsController.prompt_add_box()`
原因：同時新增 `data`、`real_data`、`box_attributes`、`listwidget`，而且可能擴充 `object_list`

2. `loadLabel()`
原因：透過 `BulkAppendBoxesCommand` 批次追加 GT，容易漏掉 attribute 與列表同步

3. `AnnotationActionsController.rename_row()` / `remove_row()` / `clear_all()`
原因：會一起碰 GT row、列表名稱、attribute 對齊、label count

4. `accept_selected_prediction()`
原因：prediction 移出後會轉進 GT，屬於跨狀態區的搬移

5. `PasteActionsController.prompt_add_candidate()` / `rename_selected()` / `remove_row()` / `clear_all()` + `PasteDocument`
原因：paste committed state 已開始集中，但 candidate state 與 UI list 還會經過這條路徑

6. `newFile()`
原因：這裡是整個 session reset 中樞，只要漏清一份 state，下一張圖就可能帶殘留資料

7. `_check_autosave_recovery()`
原因：會在 image session 啟動後回填 GT 與 attributes

## 4. 拆分前保護清單

### 這一輪已補上的自動測試

`tests/test_annotation_controller.py` 現在除了原本的 add / undo / redo 之外，還補上：

- remove box 後的 row / list / attribute 對齊
- rename box 的資料列與 `object_list` 回退
- clear all 後的完整還原
- bulk append 的批次匯入對齊

`tests/test_document.py` 也會直接保護 `AnnotationDocument` 的：

- append / remove / insert / rename
- 屬性更新與 size_tag 重算
- snapshot / restore
- metadata records 建立
- 對齊驗證

`tests/test_annotation_preview_controller.py` 與 `tests/test_annotation_workspace_controller.py` 目前也會保護：

- GT 列表選取後的高亮預覽縮放與清除
- 選取失效、清空、新檔 reset 前後的預覽重置
- 屬性面板和 preview controller 之間的接線

`tests/test_annotation_actions_controller.py` 目前也會保護：

- 手動畫框後的 class prompt 與座標換算
- 新增框取消時的畫布回復
- prediction -> GT 的 payload 建立
- GT action 入口到 command stack 的轉接

`tests/test_annotation_draw_controller.py` 目前也會保護：

- 手動畫框模式切換與 pending state 清空
- 第一點、第二點的綠色點/框繪製
- 無效第二點時的畫布回復
- 畫框完成後送進 add-box action 的座標順序

`tests/test_paste_actions_controller.py` 目前也會保護：

- paste `Add` 後的 row / list / record 同步
- paste `Add` 取消時的畫布回復
- paste `Rename` 後的 class 同步
- paste `Delete` / `Delete all` 後的平行 list 清理

`tests/test_paste_preview_controller.py` 目前也會保護：

- paste row 高亮預覽的縮放映射
- 無效 row 的預覽清除
- 手動 clear preview

`tests/test_paste_document.py` 目前也會保護：

- committed paste rows / images / records 的對齊
- rename / remove / clear 的一致性
- list identity 與 alignment 驗證

`tests/test_paste_candidate.py` 目前也會保護：

- candidate state 的 clear / clear_candidate
- placement anchor 的存在與重置

`tests/test_paste_candidate_controller.py` 目前也會保護：

- asset 載入後的 session 初始化
- horizontal flip 後的 candidate 重算
- placement click 到 preview 生成的轉接

### 每次碰主狀態時至少要手動回歸的流程

1. 開圖後新增 2 個框，確認 `Box Labels` 計數正確
2. rename 一個框，再做 undo / redo
3. delete 一個框，再做 undo / redo
4. `Load Label` 匯入一份 YOLO txt，再做一次 undo
5. `Load preds` 後測一次 `Accept` 和 `Reject`
6. 新增一筆 paste，測 rename / delete
7. 關程式後走一次 autosave recovery

## 5. 建議拆分順序

最穩的順序不是直接重寫 `MyWidget`，而是先把 truth state 包起來：

1. 先抽 `Document`，只收斂手動畫框 GT：
   - `real_data`
   - `data`
   - `box_attributes`
   - label count / class mutation helper

2. 保留 `AnnotationController`，但改成操作 `Document`

3. 把 GT 的命令入口與 UI 投影逐步抽成 controller：
   - `annotation_actions_controller.py`
   - `annotation_draw_controller.py`
   - `annotation_list_view.py`
   - `annotation_list_controller.py`
   - `annotation_preview_controller.py`
   - `annotation_workspace_controller.py`

4. 第二階段整理 paste candidate controller：
   - `chooseImg()`
   - `Hflippimg()`
   - `controlpimg()`
   - `paste()` / placement mode

目前這一步也已完成。若沒有新的產品需求或實際 bug，建議先停在這裡，避免進入低報酬拆分。

5. 最後才整理 view-only state：
   - zoom
   - tile
   - selection
   - temporary paste preview

## 6. 這一步的結論

目前最值得先保護的，不是視覺顯示，而是 `main_window.py` 內多組平行 list 的 index 對齊。只要這層先鎖住，後續把資料移到 `Document/Session` 才有機會安全地一步步落地。
