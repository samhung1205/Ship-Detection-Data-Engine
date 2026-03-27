# Paste Flow Inventory

這份文件是 `gui/main_window.py` 內 copy-paste augmentation 流程的拆分前盤點。目標不是立刻重寫 paste 功能，而是先把目前的資料形狀、同步風險與最低驗證欄杆整理清楚，避免後續抽離時只整理 UI，卻把 overlay、bbox 與 metadata 的一致性弄壞。

## 1. 目前 paste 主線的核心狀態

### A. 已提交的 paste annotation 狀態

- `pimg_data`
- `real_pimg_data`
- `paste_images`
- `paste_records`
- `pimglistwidget`
- `pimg_list`

用途：

- `pimg_data` 是目前畫布尺寸下的 paste bbox row，形狀為 `[class_name, x1, y1, x2, y2, canvas_w, canvas_h]`
- `real_pimg_data` 是原圖像素座標 row，形狀為 `[class_name, x1, y1, x2, y2]`
- `paste_images` 保存真正要疊到畫布上的 RGBA 貼圖資訊
- `paste_records` 保存匯出用的 augmentation metadata
- `pimglistwidget` / `pimg_list` 是已提交 paste state 的 UI 投影
- 目前 committed state 已先集中到 `sdde/paste_document.py` 的 `PasteDocument`，主視窗只保留 legacy 相容 property

### B. 尚未提交的 paste 編輯暫存

- `_current_asset_path`
- `pasteimg`
- `origin_pasteimg`
- `paste_canvas`
- `resizeimg`
- `rotated`
- `bc_image`
- `norm_pimg`
- `bbox_pimg`
- `real_bbox_pimg`
- `pasteimg_canvas`
- `cX` / `cY`

用途：

- 從 `Choose` 選圖到 `Add` 確認前，所有 resize / rotate / brightness / contrast / flip / placement 的中間結果都放在這裡
- 這些 state 應視為「編輯中的 candidate」，不能和已提交的 `pimg_data` / `paste_records` 混淆
- 目前 candidate state 已先集中到 `sdde/paste_candidate.py` 的 `PasteCandidateSession`

### C. paste 編輯控制 UI

- `btn_paste`
- `btn_chooseimg`
- `btn_add`
- `btn_reset`
- `Hflip`
- `slider_1` ~ `slider_4`
- `label_val_1` ~ `label_val_4`
- `pmap_pasteimg`

用途：

- `Paste Image` 進入 placement 模式
- `Choose` 載入帶 alpha 的 asset
- `Add` 把 candidate 提交成正式 paste annotation
- slider / checkbox 是 transform 參數的唯一輸入來源

## 2. 目前最重要的不變條件

### 已提交 paste annotation

- `len(pimg_data) == len(real_pimg_data) == len(paste_images) == pimglistwidget.count()`
- 在同一個 session 內，`paste_records` 應與 `real_pimg_data` 維持相同順序
- `pimg_data[i][0]`、`real_pimg_data[i][0]`、`pimglistwidget.item(i).text()`、`paste_records[i].class_name` 應該是同一個 class name

### candidate overlay

- `norm_pimg`、`bbox_pimg`、`real_bbox_pimg` 應來自同一組 transform 與 placement 計算
- `bbox_pimg` 是當前畫布座標；`real_bbox_pimg` 是原圖像素座標
- 只要 `cX / cY` 未設定，就不應嘗試生成新的 candidate overlay

### 影像 session

- `newFile()` 重置時，已提交 paste、candidate overlay、metadata、asset path 都要一起清空
- `Paste Image` 模式與 GT `Create RectBox` 模式不能同時留下 pending state

## 3. 高風險同步點

1. `chooseImg()`
原因：會初始化 asset、縮圖 preview、flip 狀態與 `PasteCandidateSession` 的 transform 起點

2. `paste()` + `controlpimg()`
原因：這裡同時碰 placement、RGBA overlay、candidate bbox、origin bbox 換算與畫布預覽，是整條 paste 流程最重的地方

3. `PasteActionsController.prompt_add_candidate()` + `PasteDocument.append_paste()`
原因：這裡會把 candidate state 提交進 `pimg_data`、`real_pimg_data`、`paste_images`、`paste_records`、`pimglistwidget`

4. `PasteActionsController.rename_selected()` / `remove_row()` / `clear_all()` + `PasteDocument.rename_paste()` / `remove_paste()` / `clear()`
原因：需要一起維護已提交 paste row、UI list 與 `paste_records`

5. `PastePreviewController.preview_row()`
原因：會從 `PasteDocument` 的已提交 row 重新映射回當前畫布高亮，容易和 GT preview 走不同規則

6. `newFile()`
原因：它是整個 paste session reset 中樞，只要漏清一份 state，下一張圖就會帶殘留 overlay 或 metadata

## 4. 已有保護與目前缺口

### 已有保護

- `tests/test_augmentation.py` 已保護：
  - `PasteRecord` 預設值
  - paste metadata JSON / CSV 匯出
  - legacy paste bbox row 解析
- `tests/test_paste_actions_controller.py` 已保護：
  - `Add` 提交後的 row / record / list count 同步
  - `Add` 取消時的畫布回復
  - `Rename` 後的 row / record class 同步
  - `Delete` / `Delete all` 後的平行 list 清理
- `tests/test_paste_preview_controller.py` 已保護：
  - paste row 高亮預覽的縮放映射
  - 無效 row 時的預覽清除
  - 手動 clear preview
- `tests/test_paste_document.py` 已保護：
  - clear / replace 的 list identity
  - append / rename / remove 的 committed state 對齊
  - alignment 驗證
- `tests/test_paste_candidate.py` 已保護：
  - candidate state 的 clear / clear_candidate
  - placement anchor state
- `tests/test_paste_candidate_controller.py` 已保護：
  - asset 載入與 thumbnail refresh
  - horizontal flip 後的 candidate 重算
  - placement click 與 candidate overlay 生成

- `docs/manual_smoke_test.md` 已包含：
  - `Choose -> Paste Image -> placement -> Add`
  - paste metadata 匯出

### 目前缺口

- paste candidate controller 目前仍直接操作 OpenCV / Qt 繪圖細節
- 真正的 GUI 整合測試仍偏少，主要依賴 unit test + 手動 smoke test

## 5. 建議拆分順序

目前五步已落地：已提交 paste 的 `Add / Rename / Delete / Delete all` 已先集中到 `gui/paste_actions_controller.py`，高亮預覽已集中到 `gui/paste_preview_controller.py`，committed state 已先集中到 `sdde/paste_document.py`，candidate / placement state 已先集中到 `sdde/paste_candidate.py`，asset / transform / placement 流程則已集中到 `gui/paste_candidate_controller.py`。後續如果沒有明確 bug 或新需求，已經可以先停在這個層級，不需要為了拆分而拆分。

1. `PasteActionsController`（已完成）
   - 已收斂 `inputPimg()` / `pimgRename()` / `pimgClear()` / `allpimgClear()`
   - 已讓提交、rename、刪除與 `paste_records` 同步走單一路徑

2. `PastePreviewController`（已完成）
   - 已收斂 `showPimg()` 與已提交 paste 的高亮預覽

3. `PasteDocument`（已完成）
   - 已收斂 `pimg_data`
   - `real_pimg_data`
   - `paste_images`
   - `paste_records`

4. `PasteCandidateSession`（已完成）
   - 已集中 `_current_asset_path`
   - `pasteimg`
   - `origin_pasteimg`
   - `paste_canvas`
   - `resizeimg`
   - `rotated`
   - `bc_image`
   - `norm_pimg`
   - `bbox_pimg`
   - `real_bbox_pimg`
   - `pasteimg_canvas`
   - `cX / cY`

5. `PasteCandidateController`（已完成）
   - 已收斂 `chooseImg()` 的 asset 載入流程
   - 已收斂 `Hflippimg()` 的 thumbnail / candidate 重算
   - 已收斂 `controlpimg()` 的 candidate overlay 計算
   - 已收斂 `paste()` / placement mode

6. 後續只在出現明確需求時再評估：
   - 更細的 OpenCV transform service 抽離
   - 更完整的 GUI 整合測試

## 6. 每次碰 paste 流程時至少要手動回歸的項目

1. `Choose` 後能正常顯示 preview 縮圖
2. `Paste Image` 後在畫布點位置，candidate overlay 能正常更新
3. 調整 `Resize / Rotate / Brightness / Contrast / HorizontalFlip` 後，candidate bbox 仍跟著變
4. `Add` 後不崩潰，且 `Paste Images` 計數正確
5. 右鍵 `Rename` 後，列表名稱與 metadata class 一致
6. 刪單筆與 `Delete all` 後，overlay 與 metadata 一起清掉
7. `Export paste metadata JSON/CSV` 可正常產生，bbox 與 class 合理
