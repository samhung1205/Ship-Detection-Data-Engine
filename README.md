# Ship Detection Data Engine (SDDE)

面向遙測 / 衛星 / 航空影像之船舶偵測研究的 **PyQt6 資料工程 GUI 工作台**。

支援多類別 HBB 標註、YOLO 格式匯入匯出、prediction overlay、error analysis、dataset QC、tile sliding-window 視圖、copy-paste augmentation、屬性標註、統計分析，以及 autosave 和 project config 持久化。

---

## 環境需求

- Python 3.10+
- 作業系統：macOS / Windows / Linux

## 安裝

```bash
pip install -r requirements.txt
```

依賴套件：

| 套件 | 版本 | 用途 |
|---|---|---|
| numpy | >=1.20.0 | 影像矩陣運算 |
| opencv-python | >=4.5.0 | 影像讀取、resize、rotation |
| PyQt6 | >=6.4.0 | GUI 框架 |
| PyYAML | >=6.0.0 | classes.yaml / project_config.yaml 讀寫 |

## 啟動

```bash
python GUI.py
```

## 重構前建議先驗證

- 第一次實際操作專案時，建議先照 [`docs/manual_smoke_test.md`](docs/manual_smoke_test.md) 跑完整手動 smoke test
- 如果要碰 `main_window.py` 的狀態拆分或同步邏輯，先看 [`docs/state_refactor_inventory.md`](docs/state_refactor_inventory.md)
- 如果準備整理 copy-paste augmentation 流程，先看 [`docs/paste_refactor_inventory.md`](docs/paste_refactor_inventory.md)
- 如果要快速回顧這一輪重構成果，請看 [`docs/refactor_summary.md`](docs/refactor_summary.md)
- 後續每次修改後，至少先跑一次 `pytest -q`
- 如果修改影響 `main_window.py`、匯入匯出、prediction、autosave 或 project config，建議再補跑 smoke test 受影響段落

---

## 操作說明

### 1. 開啟影像

- 點擊左側 **Open Image** 按鈕，或選單 **File → Open Image**（Windows / Linux：`Ctrl+O`；macOS：`Cmd+O`）
- 也可用選單 **File → Open Folder…** 載入整個資料夾，之後用 **Previous image / Next image** 逐張瀏覽
- 支援 JPG / JPEG / PNG / GIF / BMP / TIFF 格式
- 開圖後畫布自動顯示，可用 zoom in / zoom out 或拖曳 slider 縮放
- 滑鼠移到主圖上時，會出現浮動 **local zoom preview**，方便檢查小船、邊界與密集區細節

### 2. Class Mapping（類別對照）

- 點擊左側 **Class mapping** 按鈕，或選單 **File → Class mapping**（Windows / Linux：`Ctrl+I`；macOS：`Cmd+I`）
- 開啟對話框可新增、刪除、編輯類別（class_id / class_name / super_category）
- 主路徑為 `ClassMappingDialog + classes.yaml`
- 若目前有載入 `project_config.yaml`，會優先使用該 project 指定的 `classes_yaml`
- 儲存後更新目前 project 對應的 `classes.yaml`，YOLO 匯入匯出的 class index 即依此對照
- 標註 rename、paste rename、prediction accept/rewrite 都以目前 `classes.yaml` 為準；若要新增新類別，需先到 **Class mapping** 更新 catalog
- 啟動時自動從工作目錄讀取 `project_config.yaml`；若有設定 `classes_yaml`，就從該路徑載入 class catalog
- `input_window.py` 與 `data.yaml` 僅保留 legacy compatibility，不再是主流程

### 3. 標註 HBB（水平外接矩形框）

- 點擊 **Create RectBox** 按鈕進入標註模式
- 在畫布上點擊確定框的第一角，再點擊對角完成框
- 彈出對話框選擇類別名稱
- 標註會即時顯示在 Box Labels 列表和畫布上（綠色框線）
- 選取 `Box Labels` 中的既有框後，可在畫布上拖曳框內部移動，或拖曳四個角點手動縮放
- 勾選 **Hide Box** 可暫時隱藏所有框
- 右鍵點擊列表項目可 **Rename**（改類別）或 **Delete**（刪除單筆）
- **Delete all** 按鈕清除所有標註

### 4. 屬性編輯

每個標註框可設定以下研究屬性（右側面板）：

| 屬性 | 選項 | 說明 |
|---|---|---|
| size_tag | small / medium / large | 依 COCO 面積閾值自動計算，可手動改 |
| crowded | false / true | 是否在擁擠場景 |
| hard_sample | false / true | 是否作為 hard example / hard sample |
| occluded | false / true | 是否有遮擋 |
| truncated | false / true | 是否被截斷 |
| blurred | false / true | 是否模糊 |
| difficulty_tag | normal / hard / uncertain | 標註難度 |
| difficult_background | false / true | 是否屬於背景複雜的難例 |
| low_contrast | false / true | 是否屬於低對比難例 |
| scene_tag | near_shore / offshore / unknown | 場景類型 |

選擇列表中的框後，右側面板會載入該框的屬性，修改後即時同步。
其中 `hard_sample / occluded / truncated / blurred / difficult_background / low_contrast` 以 checkbox 呈現，方便快速做研究屬性與難例標記。

### 5. 載入 / 儲存 Label

- **Load Label**：選單或左側按鈕，支援：
  - YOLO HBB 格式 txt（`class_id xc yc w h`，歸一化座標）
  - COCO bbox JSON（會依目前開啟影像的檔名選出對應 image/annotations）
  - annotation metadata JSON（支援本專案匯出的 research metadata records，會還原 bbox 與屬性）
  - 預設會從 `project_config.yaml` 的 `label_root` 開始選檔
  - 載入前可選擇 **Append GT** 或 **Replace GT**
    - `Append GT`：把載入結果追加到目前 GT 標註
    - `Replace GT`：用載入結果覆蓋目前 GT 標註（paste annotations 不受影響）
- **Save Label**：彈出對話框，選擇格式：
  - **YOLO(v5~10)**：歸一化中心座標 `class_id xc yc w h`
  - **Bounding Boxes**：絕對座標 `class_id x1 y1 x2 y2`
  - **COCO JSON**：單張影像的 `images / annotations / categories` 結構，bbox 為 `[x, y, w, h]`
  - **Pascal VOC XML**：單張影像的 `annotation/object/bndbox` XML
- `Save Label` 會把目前影像中的 **GT + committed paste annotations** 一起輸出
- `Save Label` 對話框的預設格式會跟 `project_config.yaml` 的 `default_export_format` 一致
- 若目前影像位於 `image_root` 之下，預設輸出路徑會映射到 `label_root` 對應位置，例如 `images/train/a.jpg -> labels/train/a.txt`
- **Show Label**：在新視窗中預覽當前標註疊圖

### 6. Prediction Overlay（模型預測疊圖）

- **Load preds**（左側按鈕）或選單 **File → Load predictions…**
- 讀取 YOLO 格式 txt，支援第 6 欄 confidence：`class_id xc yc w h [conf]`
- 若目前影像位於 `images/` 資料夾，prediction 檔案選擇器會優先從相鄰的 `predictions/` 資料夾開啟，例如 `dataset/test/images` → `dataset/test/predictions`
- 也可用選單 **File → Load prediction folder…** 指定一個 prediction sidecar 資料夾；之後在資料夾瀏覽模式下切換影像時，會自動讀取同 basename 的 `.txt` prediction 檔
- 會建立 session 內的 review queue 狀態：`pending / partial / reviewed`
- 下方狀態列會顯示目前 queue 摘要，例如 pending 數量與當前影像的 accepted / rejected / remaining
- 選單 **File → Next review image** 可直接跳到下一張**尚未 reviewed**且有 prediction sidecar 的影像
- review queue 會以「目前 image folder + prediction folder」為 key 持久化到 sidecar state；重新載入同一組資料夾時，可選擇 **Resume review** 或 **Start fresh**
- 選單 **File → Clear saved review state…** 可清除目前這組 image folder + prediction folder 的已存 review 狀態，並立刻用 sidecar predictions 重建一輪新的 queue
- 選單 **Analysis → Prediction review summary…** 可對 **Current folder** 或 **Current project** 產生 review 報表
  - 若目前已載入 prediction folder，會直接使用該路徑
  - 若尚未載入 prediction folder，才會要求選擇
  - 結果會顯示 `reviewed / partial / pending / no_predictions` 的影像數，以及 accepted / rejected / remaining prediction 總數
  - **Export CSV…** / **Export JSON…** 可匯出逐影像 review summary，方便做 project-level review 進度追蹤
- 也可用選單 **File → Load YOLO model…** 載入模型，再用左側 **Run model** 按鈕或 **File → Run model prediction** 直接對當前影像推論
- 模型直推論目前為 optional 功能，需要額外安裝 `ultralytics` 與 `torch`
- 預測框以**橘色虛線**顯示在畫布上（與 GT 綠色實線區分）
- 勾選 **Show preds** 控制是否顯示
- 右上方 **Pred conf >=** slider 可調整顯示門檻（`0.00` ~ `1.00`），同步影響 prediction list、畫布 overlay 與 error analysis 使用的 prediction 集合
- 選取左側 prediction list 中的一筆後，可在畫布上直接拖曳移動或拖四角調整大小；被修改過的 prediction 會在列表中標示為 **`[edited]`**
- prediction list 右鍵可 **Change class…**，先修正類別再決定是否接受
- 左側預測列表中選取一筆後：
  - **Accept**：轉為正式 GT 標註（綠色框），從預測列表移除
  - **Reject**：刪除該筆預測
- 選單 **File → Accept all visible predictions** / **Reject all visible predictions** 可對目前 conf threshold 下可見的 prediction 批次處理
- 當當前影像的 predictions 全部處理完畢並進入 `reviewed` 狀態時，review queue 會自動跳到下一張待審影像
- 選單 **File → Clear predictions** 清空所有預測

### 7. Error Analysis（誤差分析）

- 選單 **Analysis → Run error analysis…**
- 可選 scope：
  - **Current image**：分析目前開啟影像的 GT 與目前載入的 predictions
  - **Current folder**：掃描目前影像所在資料夾內的所有支援影像，並要求指定一個 **prediction folder**；若目前 project config 可解析 `image_root`，會先找 `prediction_root/<image 相對路徑>/<basename>.txt`，找不到再 fallback 到 `prediction_root/<basename>.txt`
  - **Current project**：若已載入 `project_config.yaml`，會以 `image_root` 為遞迴掃描範圍，並依 `label_root` / 選定的 `prediction root` 以相對路徑配對
- 選單 **Analysis → Show GT/Pred IoU overlay** 可把 IoU 配對結果直接疊在主畫布上：
  - 已配對 GT / prediction 之間會畫出連線並標示 `TP / WrongClass / Localization / Duplicate` 與 IoU
  - 未配對 prediction 會在框旁標示 `FP`
  - 未配對 GT 會在框旁標示 `FN`
- 自動以 greedy IoU matching 比對當前 GT 與 predictions
- 顯示表格：Type / IoU / GT class / Pred class / Confidence
- 可用 **Type** 下拉選單快速篩選 `TP / FP / FN / WrongClass / Localization / Duplicate`
- 可勾選 **Bookmarked only** 只瀏覽已收藏案例
- 若 GT 有屬性標註，還可依 **Size / Scene / Difficulty / Crowded / Hard sample / Occluded / Truncated / Blurred** 篩選錯誤案例，方便分析 small ship、near_shore、hard sample、遮擋或模糊樣本等研究情境
- 表格也會直接顯示這些 GT 屬性欄位，方便快速比對 error case 與資料特性
- 分類規則：

| 類型 | 條件 |
|---|---|
| TP | IoU ≥ 0.5 且類別相同 |
| WrongClass | IoU ≥ 0.5 且類別不同 |
| Localization | 0.1 ≤ IoU < 0.5 |
| Duplicate | pred 重疊已匹配的 GT |
| FP | 未匹配的 pred |
| FN | 未匹配的 GT |

- 可勾選 **Bookmark**、填寫 **Notes**
- 大量 folder / project cases 時，結果表格會限制實際渲染的前幾千筆，避免 Qt table 一次建立過多 row 造成記憶體暴增；上方 summary 仍會顯示完整 case 數
- **Export CSV…** 仍會匯出所有錯誤案例，不只匯出表格目前顯示的前幾千筆
- committed paste annotations 也會納入目前影像的 GT 集合一起分析，避免與實際匯出的 label 不一致
- `Current folder` 也會顯示 **Image** 欄位，方便在多張圖之間追 case
- `Current folder` 只有在目前影像真的有 live GT / committed paste，或已載入並審核過的 live prediction 狀態時，才會用 GUI 內的狀態覆蓋該張圖；若目前 GUI 狀態是空的，會回到磁碟上的 label / prediction sidecar，避免 summary 把已存在的檔案誤算成 missing
- `Current folder` 會顯示 **Predictions matched** 與 **Label root**，方便確認實際掃描到多少 prediction sidecar，以及 GT label 是從哪個目錄解析而來
- 在真正打開 folder error analysis 結果前，會先顯示一個 summary，列出 `Images found / Labels matched / Predictions matched / Images analyzable / Confidence threshold`
- `Current project` 也會在真正打開結果前顯示相同型態的 summary，但 scope 會改成整個 `image_root`
- 若選到的 prediction folder 沒有任何同 basename 的 `.txt` prediction sidecar，會先提示你結果只會使用 GT labels
- **Analysis → Start FP-to-label review…** 可直接用 folder / project error analysis 的 `FP` rows 建立補標 queue：
  - 先選 scope 與 prediction folder，GUI 會依目前 `Pred conf >=` 門檻掃描 FP candidates
  - 建立 queue 後會跳到第一個 FP 影像並選中對應 prediction
  - **Analysis → Next FP** 跳到下一個 FP candidate；若下一個 candidate 仍在同一張影像，不會重新載圖，避免清掉尚未儲存的補標
  - 可沿用 prediction list 的 **Change class… / Accept / Reject**；確認漏標後按 **Accept** 轉正式 GT，再補屬性並 **Save Label**
  - FP review 期間不會觸發原本 prediction review 的自動跳下一張，避免干擾補標 queue

### 8. Dataset Statistics（統計分析）

- 選單 **Analysis → Dataset statistics…**
- 可選 scope：
  - **Current image**：分析目前開啟影像的 **GT + committed paste annotations**
  - **Current folder**：掃描目前影像所在資料夾內的所有支援影像，按 basename 尋找對應 label sidecar
  - **Current project**：若已載入 `project_config.yaml`，會以 `image_root` 為遞迴掃描範圍，並依 `label_root` 以相對路徑配對 GT labels
- 多個分頁：
  - **Overview**：總影像數、總標註數、平均每張圖標註數、類別數
  - **Class dist**：各類別標註數量
  - **Size dist**：small / medium / large 分布
  - **Scene dist**：near_shore / offshore / unknown 的數量與比例
  - **Bbox stats**：寬 / 高 / 面積 / 長寬比的 min / max / mean / std
    - 若 metadata 中有 `rotation_deg` / `angle_deg`，也會一併統計旋轉角度分布
  - **Class × Size**：類別 × 尺度交叉表
- **Export JSON…** / **Export CSV…** 匯出統計結果
- Overview 也會顯示 **Labeled images / Unlabeled images**
- `Current folder` 目前優先讀取與影像同 basename 的 `.json` / `.txt` label sidecar；若 `project_config.yaml` 中有設定 `image_root / label_root`，也會嘗試依相對路徑對應到 label 目錄
- 若沒有 `project_config.yaml`，但目前資料夾結構符合常見的 `images/` 與 sibling `labels/` 佈局，也會自動推斷 GT label 位置
- folder-level statistics / error analysis 讀影像尺寸時會優先只讀常見格式（BMP/JPEG/PNG/GIF/TIFF）的檔頭，不會為了拿寬高而完整解碼整張影像，避免大量 BMP 掃描時記憶體暴增
- 在真正打開 folder statistics 結果前，會先顯示一個 summary，列出 `Images found / Labels matched / Annotations found`
- `Current project` 也會先顯示同型態的 summary，但 scope 會改成整個 `image_root`
- 若目前開啟影像有尚未另存的新標註、committed paste，或曾載入後被清空的標註狀態，`Current folder` 會以當前 GUI 內的 live annotations 覆蓋該張圖的磁碟 label；若目前沒有 live annotation override，則會讀取磁碟上的 label sidecar

### 9. Dataset QC（批次驗證）

- 選單 **Analysis → Dataset QC…**
- 可選 scope：
  - **Current folder**：掃描目前影像所在資料夾內的所有支援影像
  - **Current project**：若已載入 `project_config.yaml`，會以 `image_root` 為遞迴掃描範圍
- 執行前可選擇：
  - **Include predictions…**：額外指定 prediction folder，一起檢查 prediction sidecar
  - **Labels only**：只檢查 GT labels
- 會先顯示 summary，列出：
  - `Images found`
  - `Labels matched`
  - `Predictions matched`
  - `Issues found`
- 結果表格會列出：
  - `Image`
  - `Source`（label / prediction）
  - `Issue type`
  - `Line`
  - `Detail`
  - `File`
- 結果視窗另提供：
  - **Overview**：總影像數、matched labels / predictions、clean images、images with issues
  - **Issue types**：各 issue type 的數量與受影響影像數
- 目前內建檢查：
  - missing label / missing prediction
  - empty label / empty prediction
  - YOLO 欄位數錯誤
  - 無法解析數值
  - 非法 class id
  - normalized center / width / height 超出範圍
  - bbox 超出 `[0, 1]`
  - prediction confidence 超出 `[0, 1]`
  - 無法讀取或不支援的 label 格式
- **Export CSV…** 可匯出所有逐筆 issue
- **Export Summary JSON…** 可匯出聚合 summary，方便做 dataset health 檢查、研究紀錄與後續自動化報表

### 10. Tile / Sliding Window（分格檢視）

- 左下角 **Tile view** 面板
- 設定 **Size**（tile 邊長，預設 640）和 **Stride**（步幅，預設 480）
- 勾選 **Enable** 啟用分格視圖：
  - 畫布上顯示綠色 tile 邊框，tile 外區域暗化
  - 方向箭頭可切換上下左右鄰近 tile，index 標籤顯示 `3 / 16`
  - 也可用 `Alt + Arrow` 快速切換目前 tile 的上下左右鄰居
  - 勾選 **Overview** 可回到全圖查看 tile 分布，並高亮目前 tile；在 overview 上直接點選某個 tile 可跳入該 tile
  - Tile 面板會提示 **Boundary** 數量；若顯示 `Boundary: 2`，代表目前 tile 內有 2 個 GT 框跨越 tile 邊界，需留意重複標註或截斷漏標
- tile 只是視角，所有標註始終寫回全圖座標，切換 tile 不會造成框漂移

### 11. Copy-Paste（圖像貼上）

- 點擊 **Paste Image** 按鈕進入貼圖模式
- 右下方 **Choose Image** 選擇 PNG asset（含 alpha 透明通道）
- 可在 **Mode** 切換 `Manual / Smart zone`；`Smart zone` 會要求先用 **Set zone** 在畫布上框出可貼圖區域，只有完整落在該區域內的 paste 才能 **Add**
- **Target** 可選 `small / medium / large`，右側會顯示目前 zoom/export 比例下的建議縮放範圍與目前 paste 尺度
- 用 7 個 slider 調整：Resize / Rotation / Brightness / Contrast / Blur / Opacity / Feather
- **Effects (n)** 可開啟進階效果對話框，額外設定 `Shadow` 與 `Motion Blur`，避免主畫面右下角再堆更多 slider
- 勾選 **HorizontalFlip** / **VerticalFlip** 控制翻轉
- 在畫布上點擊放置，自動偵測 alpha 邊界產生 bbox
- 若最終 alpha 幾乎為空（例如 opacity 太低），預覽仍會更新，但 **Add** 會保持 disabled，避免提交空貼圖
- 每次貼圖自動記錄 `PasteRecord`（含 asset 路徑、flip / blur / opacity / feather 等 transform 參數、bbox、timestamp）
- 選單 **File → Export paste metadata (JSON/CSV)** 匯出完整操作紀錄
- 一旦 **Add** 成功，該 paste annotation 會視為正式可匯出標註，並納入 metadata export / statistics / error analysis
- **Save Image** 會以原圖解析度重建合成結果，不會直接把當前 GUI 縮放畫面存出去；若想避免 JPEG 再壓縮，建議存成 PNG / BMP

### 12. Metadata Export（屬性匯出）

- 選單 **File → Export annotation metadata (JSON)…** / **(CSV)…**
- 匯出每筆標註的完整欄位：image_path / class_id / class_name / super_category / x1 / y1 / x2 / y2 / size_tag / crowded / hard_sample / occluded / truncated / blurred / difficulty_tag / difficult_background / low_contrast / scene_tag / annotation_source
- `annotation_source` 會標記來源為 `gt` 或 `paste`
- metadata export 目前會把 **GT + committed paste annotations** 一起匯出

### 13. Undo / Redo

- 選單 **Edit → Undo**（Windows / Linux：`Ctrl+Z`；macOS：`Cmd+Z`） / **Redo**（Windows / Linux：`Ctrl+Shift+Z` 或 `Ctrl+Y`；macOS：`Cmd+Shift+Z`）
- 最多保留 50 步操作紀錄
- 支援：新增框、刪除框、清除全部、重新命名、批次載入 label

### 14. Autosave（自動存檔）

- 開圖後自動啟動定時器（預設 60 秒），週期性存檔到 `.autosave/` 目錄
- 存檔格式：`<圖片名>.autosave.json`（含 real_data + box_attributes + object_list）
- 下次開同一張圖時，若偵測到 autosave 檔，自動詢問是否恢復
- 手動 Save Label 成功後自動清除 autosave 檔
- 狀態列顯示 "Autosaved" / "Restored from autosave" / "Saved (autosave cleared)"

### 15. Project Config（專案設定）

- 選單 **File → Open project config…** / **Save project config…**
- 設定檔格式為 `project_config.yaml`，包含：

```yaml
project_name: ship_detection_project
project_root: "."
image_root: dataset
label_root: dataset
classes_yaml: classes.yaml
default_export_format: yolo_hbb
autosave_seconds: 60
recent_images:
  - /path/to/last_opened.jpg
tile_size: 640
tile_stride: 480
```

- 啟動時自動讀取工作目錄下的 `project_config.yaml`（如果存在）
- 開圖時自動加入 `recent_images`
- `project_config.yaml` 現在會實際控制：
  - `classes_yaml`：啟動時 class catalog 的載入位置，以及 class mapping dialog 的預設讀寫位置
  - `image_root`：**Open Image** / **Open Folder** 的預設起始目錄
  - `label_root`：**Load Label** 的預設起始目錄，以及 **Save Label** 的預設輸出根目錄
  - `default_export_format`：**Save Label** 對話框的預設格式
  - `autosave_seconds` / `tile_size` / `tile_stride`
- `project_root` 與其他相對路徑會以 `project_config.yaml` 所在位置解析，因此每個研究專案都可以攜帶自己的一套 classes / images / labels / export defaults

### 快捷鍵總覽

| 快捷鍵 | 功能 |
|---|---|
| `W` | Create RectBox |
| `D` | Delete selected box |
| `S` | Save Label |
| `H` | Hide / Show boxes |
| `Ctrl+O` / `Cmd+O` | Open Image |
| `Ctrl+I` / `Cmd+I` | Class mapping |
| `Ctrl+L` / `Cmd+L` | Load Label |
| `Ctrl+Z` / `Cmd+Z` | Undo |
| `Ctrl+Shift+Z` / `Ctrl+Y` / `Cmd+Shift+Z` | Redo |
| `Alt+Arrow` | Tile 上下左右導航 |
| `Alt+G` | Toggle tile overview |

---

## 專案結構

```
GUI/
├── GUI.py                          # 程式進入點
├── requirements.txt                # Python 依賴
├── classes.yaml                    # 類別定義檔（啟動時自動讀取）
├── data.yaml                       # legacy compatibility only（舊 YOLO data.yaml 路徑）
├── project_config.yaml             # 專案設定檔（可選，啟動時自動讀取）
├── docs/
│   ├── manual_smoke_test.md        #   重構前後共用的手動回歸清單
│   ├── paste_refactor_inventory.md #   copy-paste augmentation 狀態盤點與拆分前保護清單
│   ├── refactor_summary.md         #   本輪重構成果摘要 / 交接筆記
│   └── state_refactor_inventory.md #   main_window 狀態盤點與拆分前保護清單
│
├── gui/                            # GUI 層（PyQt6 widgets + controllers）
│   ├── __init__.py                 #   匯出 MyWidget
│   ├── main_window.py              #   主視窗（MyWidget），負責 orchestration 與剩餘 UI glue
│   ├── canvas_widget.py            #   ImageCanvasWidget — 可捲動畫布、縮放重繪
│   ├── canvas_utils.py             #   繪圖函式：bbox / paste / prediction / tile overlay
│   ├── annotation_controller.py    #   Undo/Redo command stack（Add/Remove/Rename/Clear/Bulk）
│   ├── annotation_actions_controller.py # GT add / rename / delete / clear action entry controller
│   ├── annotation_draw_controller.py # GT canvas drawing interaction controller
│   ├── annotation_list_view.py     #   GT list / count 的 UI 投影 adapter
│   ├── annotation_list_controller.py #  GT list selection / context menu controller
│   ├── annotation_preview_controller.py # GT row highlight preview controller
│   ├── annotation_workspace_controller.py # GT selection + attr-panel sync controller
│   ├── paste_candidate_controller.py # Paste asset load / transform / placement controller
│   ├── paste_actions_controller.py #   Paste add / rename / delete / clear action entry controller
│   ├── paste_preview_controller.py #   Paste row highlight preview controller
│   ├── attribute_panel.py          #   右側屬性編輯面板（size / crowded / difficulty / hard-case flags / scene）
│   ├── class_mapping_service.py    #   classes.yaml 讀寫 + ClassCatalog 管理
│   ├── tile_panel.py               #   左下方 Tile view 面板（Size/Stride/Enable/Nav）
│   ├── constants.py                #   共用 UI 樣式常數
│   │
│   └── dialogs/                    #   對話框視窗
│       ├── __init__.py             #     匯出所有 dialog
│       ├── class_mapping_dialog.py #     Class mapping 編輯對話框
│       ├── error_analysis_dialog.py#     Error analysis 結果表格 + CSV 匯出
│       ├── prediction_review_report_dialog.py # Prediction review folder/project 報表
│       ├── statistics_dialog.py    #     Dataset statistics 四分頁表格 + JSON/CSV 匯出
│       ├── validation_dialog.py    #     Dataset QC issue table + CSV 匯出
│       ├── input_window.py         #     legacy compatibility only（不屬於主流程）
│       ├── showlab_window.py       #     Show Label 預覽視窗
│       ├── saveimg_window.py       #     Save Image 對話框
│       └── savelab_window.py       #     Save Label 格式選擇對話框
│
├── sdde/                           # 資料模型與服務層（純 Python，不依賴 PyQt6）
│   ├── __init__.py                 #   統一匯出所有 public API
│   ├── models.py                   #   核心資料模型：HBBBoxPx / HBBBoxYoloNorm / HBBAnnotation / ImageAnnotation
│   ├── document.py                 #   GT annotation Document 骨架（集中 legacy 平行 lists）
│   ├── config.py                   #   legacy data.yaml 相容解析
│   ├── class_catalog.py            #   ClassCatalog / ClassInfo — 類別目錄管理
│   ├── classes_yaml.py             #   classes.yaml 格式讀寫
│   ├── import_export.py            #   YOLO HBB parse / export、bbox txt export
│   ├── legacy_rows.py              #   legacy GUI row 格式 <-> HBBAnnotation 轉接
│   ├── metadata_export.py          #   per-image annotation metadata → JSON / CSV
│   ├── dataset_scan.py             #   folder / dataset label 掃描與 records 聚合
│   ├── attributes.py               #   研究屬性欄位定義、size_tag 計算（COCO 閾值）
│   ├── prediction.py               #   PredictionRecord、YOLO prediction 解析（含 confidence）
│   ├── prediction_scan.py          #   prediction sidecar path / folder review helper
│   ├── prediction_review.py        #   session-level review queue state / summary helper
│   ├── prediction_review_report.py #   folder/project review summary + CSV/JSON helper
│   ├── prediction_review_store.py  #   review queue persistence helper（resume / fresh / clear）
│   ├── model_inference.py          #   optional YOLO model 載入 / 單張影像推論 helper
│   ├── error_analysis.py           #   IoU 計算、GT-vs-pred greedy matching、ErrorCase、CSV 匯出
│   ├── statistics.py               #   dataset 統計分析（class / size / bbox distributions）、JSON/CSV
│   ├── error_analysis_scan.py      #   folder-level GT/pred 掃描與 ErrorCase 聚合
│   ├── validation.py               #   dataset / project batch QC helper（label/prediction sidecar 檢查）
│   ├── tile.py                     #   TileConfig / TileRect / compute_tile_grid / 座標轉換
│   ├── augmentation.py             #   PasteRecord（copy-paste transform 紀錄）、JSON/CSV 匯出
│   ├── paste_candidate.py          #   in-progress paste candidate session（暫存 transform / placement state）
│   ├── paste_document.py           #   committed paste state Document（集中 legacy 平行 lists）
│   ├── project_config.py           #   ProjectConfig dataclass、YAML load / save
│   └── autosave.py                 #   autosave write / read / remove（JSON sidecar）
│
└── tests/                          # 單元測試（pytest）
    ├── test_models.py              #   HBBBoxPx / HBBAnnotation 測試
    ├── test_config.py              #   data.yaml 相容解析測試
    ├── test_import_export.py       #   YOLO parse / export round-trip 測試
    ├── test_legacy_rows.py         #   legacy row 與 HBBAnnotation 轉接測試
    ├── test_classes_yaml.py        #   classes.yaml 讀寫 round-trip
    ├── test_annotation_controller.py #  Undo / Redo command stack 測試
    ├── test_annotation_actions_controller.py # GT action entry controller 測試
    ├── test_annotation_draw_controller.py # GT canvas drawing controller 測試
    ├── test_annotation_list_view.py #  GT list / count projection 測試
    ├── test_annotation_list_controller.py # GT list interaction controller 測試
    ├── test_annotation_preview_controller.py # GT highlight preview controller 測試
    ├── test_annotation_workspace_controller.py # GT workspace controller 測試
    ├── test_paste_candidate_controller.py # Paste candidate controller 測試
    ├── test_paste_actions_controller.py # Paste action entry controller 測試
    ├── test_paste_preview_controller.py # Paste highlight preview controller 測試
    ├── test_paste_candidate.py     #   Paste candidate session 測試
    ├── test_paste_document.py      #   Paste document API / 對齊保護測試
    ├── test_document.py            #   GT document API / 對齊保護測試
    ├── test_attributes.py          #   size_tag 計算、default_attributes 測試
    ├── test_metadata_export.py     #   build_annotation_records + JSON/CSV 匯出
    ├── test_dataset_scan.py        #   folder label 掃描、image_root→label_root 與 images→labels 推斷
    ├── test_prediction.py          #   YOLO prediction 解析（含/不含 confidence）
    ├── test_prediction_scan.py     #   prediction sidecar path / folder review helper
    ├── test_prediction_review.py   #   review queue state / status / summary helper
    ├── test_prediction_review_report.py # folder/project review summary 掃描與匯出
    ├── test_prediction_review_store.py # review queue persistence round-trip / delete
    ├── test_error_analysis.py      #   IoU / matching / error_type / CSV 匯出
    ├── test_error_analysis_scan.py #   folder-level GT/pred 掃描、prediction match 計數與 current-image override
    ├── test_statistics.py          #   dataset stats 計算 + JSON/CSV 匯出
    ├── test_validation.py          #   dataset/project QC helper 測試
    ├── test_tile.py                #   tile grid 計算 / 座標轉換 / annotations_in_tile
    ├── test_augmentation.py        #   PasteRecord / JSON/CSV 匯出
    ├── test_project_config.py      #   ProjectConfig YAML round-trip
    └── test_autosave.py            #   autosave write / read / remove
```

---

## 執行測試

```bash
pip install pytest
pytest -q
```

目前共 264 個測試，涵蓋所有 `sdde/` 模組的核心邏輯、legacy `data.yaml` 相容解析、GUI 與 service layer 之間的 row adapter、annotation controller 的狀態對齊保護、GT / paste document API、paste candidate session，以及 GT 與 paste 的 action / draw / list / preview / workspace / candidate controllers，其中也包含 paste payload 在含有 Qt `QImage` 物件時的提交回歸保護、`Load Label` 的 append/replace 路徑、GT + paste 的聚合 export / analysis 行為、`Current folder` 與 `Current project` statistics 的 label 掃描、preflight summary 與 project-style 路徑對應、未載入 project config 時的 `images/ -> labels/` 自動推斷、folder/project-level 的輕量 header size probe、`Current folder` / `Current project` error analysis 的 GT/pred 掃描、label/prediction match 計數 / preflight summary / current-image override、`Current folder` / `Current project` dataset QC 的 missing sidecar / invalid YOLO / out-of-range bbox 檢查與 summary export、folder/project-level 的 prediction review summary / CSV / JSON 匯出，以及 `project_config` 對 classes/image/label/export default 的路徑接線，還有 folder-level prediction review / batch accept-reject / review queue 的 sidecar workflow 與 persistence。
如果準備進行重構，請再搭配 [`docs/manual_smoke_test.md`](docs/manual_smoke_test.md) 的手動 smoke test 清單一起回歸。

---

## 架構設計原則

1. **資料與 UI 分離** — `sdde/` 是純 Python 資料層，不依賴 PyQt6；`gui/` 是 UI 層，呼叫 `sdde/` 的 API
2. **Class mapping 單一路徑** — 主流程使用 `ClassMappingDialog + classes.yaml`；`InputWindow + data.yaml` 只保留 legacy compatibility
3. **Command pattern** — 所有標註操作透過 `annotation_controller.py` 的 command stack，支援 Undo/Redo
4. **Label IO 單一來源** — GUI 載入/儲存 label 會經過 `sdde/import_export.py`，避免 UI 與 service layer 各自維護一份格式邏輯
5. **Tile 是視角不是資料** — tile 模式只影響畫布顯示，所有 annotation 始終寫回全圖座標
6. **Prediction 是獨立資料層** — prediction 疊在 GT 之上顯示，Accept 才轉為正式標註
7. **Autosave 是 sidecar** — 存在 `.autosave/` 目錄下的 JSON，不干擾原始 label 檔
8. **Paste state 分層收斂** — 已提交的 paste state 已集中到 `sdde/paste_document.py`，編輯中的 candidate / placement state 已集中到 `sdde/paste_candidate.py`，並由 `paste_candidate_controller.py` / `paste_actions_controller.py` / `paste_preview_controller.py` 操作，避免 `main_window.py` 直接持有過多平行欄位

---

## 檔案格式說明

### classes.yaml

```yaml
project_name: ship_detection_project
classes:
  - class_id: 0
    class_name: naval
    super_category: vessel
  - class_id: 1
    class_name: merchant
    super_category: vessel
```

### YOLO HBB Label（歸一化）

```
0 0.5125 0.3200 0.0850 0.1200
1 0.7800 0.6500 0.1200 0.0900
```

每行：`class_id x_center y_center width height`（相對於圖片寬高的 0~1 值）

### YOLO Prediction（含 confidence）

```
0 0.5125 0.3200 0.0850 0.1200 0.87
1 0.7800 0.6500 0.1200 0.0900 0.62
```

第 6 欄為信心分數（可略，預設 1.0）

### project_config.yaml

```yaml
project_name: ship_detection_project
project_root: "."
image_root: dataset
label_root: dataset
classes_yaml: classes.yaml
default_export_format: yolo_hbb
autosave_seconds: 60
recent_images: []
tile_size: 640
tile_stride: 480
```

---

## License

Internal research tool — 僅供研究用途。
