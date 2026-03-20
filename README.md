# Ship Detection Data Engine (SDDE)

面向遙測 / 衛星 / 航空影像之船舶偵測研究的 **PyQt6 資料工程 GUI 工作台**。

支援多類別 HBB 標註、YOLO 格式匯入匯出、prediction overlay、error analysis、tile sliding-window 視圖、copy-paste augmentation、屬性標註、統計分析，以及 autosave 和 project config 持久化。

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

---

## 操作說明

### 1. 開啟影像

- 點擊左側 **Open Image** 按鈕，或選單 **File → Open Image** (`Ctrl+O`)
- 支援 JPG / PNG / GIF / BMP 格式
- 開圖後畫布自動顯示，可用 zoom in / zoom out 或拖曳 slider 縮放

### 2. Class Mapping（類別對照）

- 點擊左側 **Class mapping** 按鈕，或選單 **File → Class mapping** (`Ctrl+I`)
- 開啟對話框可新增、刪除、編輯類別（class_id / class_name / super_category）
- 儲存後更新 `classes.yaml`，YOLO 匯入匯出的 class index 即依此對照
- 啟動時自動從工作目錄讀取 `classes.yaml`

### 3. 標註 HBB（水平外接矩形框）

- 點擊 **Create RectBox** 按鈕進入標註模式
- 在畫布上點擊確定框的第一角，再點擊對角完成框
- 彈出對話框選擇類別名稱
- 標註會即時顯示在 Box Labels 列表和畫布上（綠色框線）
- 勾選 **Hide Box** 可暫時隱藏所有框
- 右鍵點擊列表項目可 **Rename**（改類別）或 **Delete**（刪除單筆）
- **Delete all** 按鈕清除所有標註

### 4. 屬性編輯

每個標註框可設定以下研究屬性（右側面板）：

| 屬性 | 選項 | 說明 |
|---|---|---|
| size_tag | small / medium / large | 依 COCO 面積閾值自動計算，可手動改 |
| crowded | false / true | 是否在擁擠場景 |
| difficulty_tag | normal / hard / uncertain | 標註難度 |
| scene_tag | near_shore / offshore / unknown | 場景類型 |

選擇列表中的框後，右側面板會載入該框的屬性，修改後即時同步。

### 5. 載入 / 儲存 Label

- **Load Label**：選單或左側按鈕，讀取 YOLO HBB 格式 txt（`class_id xc yc w h`，歸一化座標）
- **Save Label**：彈出對話框，選擇格式：
  - **YOLO(v5~10)**：歸一化中心座標 `class_id xc yc w h`
  - **Bounding Boxes**：絕對座標 `x1 y1 x2 y2 class_id`
- **Show Label**：在新視窗中預覽當前標註疊圖

### 6. Prediction Overlay（模型預測疊圖）

- **Load preds**（左側按鈕）或選單 **File → Load predictions…**
- 讀取 YOLO 格式 txt，支援第 6 欄 confidence：`class_id xc yc w h [conf]`
- 預測框以**橘色虛線**顯示在畫布上（與 GT 綠色實線區分）
- 勾選 **Show preds** 控制是否顯示
- 左側預測列表中選取一筆後：
  - **Accept**：轉為正式 GT 標註（綠色框），從預測列表移除
  - **Reject**：刪除該筆預測
- 選單 **File → Clear predictions** 清空所有預測

### 7. Error Analysis（誤差分析）

- 選單 **Analysis → Run error analysis…**
- 自動以 greedy IoU matching 比對當前 GT 與 predictions
- 顯示表格：Type / IoU / GT class / Pred class / Confidence
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
- **Export CSV…** 匯出所有錯誤案例

### 8. Dataset Statistics（統計分析）

- 選單 **Analysis → Dataset statistics…**
- 四個分頁：
  - **Class dist**：各類別標註數量
  - **Size dist**：small / medium / large 分布
  - **Bbox stats**：寬 / 高 / 面積 / 長寬比的 min / max / mean / std
  - **Class × Size**：類別 × 尺度交叉表
- **Export JSON…** / **Export CSV…** 匯出統計結果

### 9. Tile / Sliding Window（分格檢視）

- 左下角 **Tile view** 面板
- 設定 **Size**（tile 邊長，預設 640）和 **Stride**（步幅，預設 480）
- 勾選 **Enable** 啟用分格視圖：
  - 畫布上顯示綠色 tile 邊框，tile 外區域暗化
  - `<` `>` 按鈕切換 tile，index 標籤顯示 `3 / 16`
- tile 只是視角，所有標註始終寫回全圖座標，切換 tile 不會造成框漂移

### 10. Copy-Paste（圖像貼上）

- 點擊 **Paste Image** 按鈕進入貼圖模式
- 右下方 **Choose Image** 選擇 PNG asset（含 alpha 透明通道）
- 用 4 個 slider 調整：Resize / Rotation / Brightness / Contrast
- 勾選 **HorizontalFlip** 水平翻轉
- 在畫布上點擊放置，自動偵測 alpha 邊界產生 bbox
- 每次貼圖自動記錄 `PasteRecord`（含 asset 路徑、transform 參數、bbox、timestamp）
- 選單 **File → Export paste metadata (JSON/CSV)** 匯出完整操作紀錄

### 11. Metadata Export（屬性匯出）

- 選單 **File → Export annotation metadata (JSON)…** / **(CSV)…**
- 匯出每筆標註的完整欄位：image_path / class_id / class_name / super_category / x1 / y1 / x2 / y2 / size_tag / crowded / difficulty_tag / scene_tag

### 12. Undo / Redo

- 選單 **Edit → Undo** (`Ctrl+Z`) / **Redo** (`Ctrl+Shift+Z`)
- 最多保留 50 步操作紀錄
- 支援：新增框、刪除框、清除全部、重新命名、批次載入 label

### 13. Autosave（自動存檔）

- 開圖後自動啟動定時器（預設 60 秒），週期性存檔到 `.autosave/` 目錄
- 存檔格式：`<圖片名>.autosave.json`（含 real_data + box_attributes + object_list）
- 下次開同一張圖時，若偵測到 autosave 檔，自動詢問是否恢復
- 手動 Save Label 成功後自動清除 autosave 檔
- 狀態列顯示 "Autosaved" / "Restored from autosave" / "Saved (autosave cleared)"

### 14. Project Config（專案設定）

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

### 快捷鍵總覽

| 快捷鍵 | 功能 |
|---|---|
| `Ctrl+O` | Open Image |
| `Ctrl+I` | Class mapping |
| `Ctrl+L` | Load Label |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |

---

## 專案結構

```
GUI/
├── GUI.py                          # 程式進入點
├── requirements.txt                # Python 依賴
├── classes.yaml                    # 類別定義檔（啟動時自動讀取）
├── project_config.yaml             # 專案設定檔（可選，啟動時自動讀取）
│
├── gui/                            # GUI 層（PyQt6 widgets + controllers）
│   ├── __init__.py                 #   匯出 MyWidget
│   ├── main_window.py              #   主視窗（MyWidget），所有 UI 邏輯的中樞
│   ├── canvas_widget.py            #   ImageCanvasWidget — 可捲動畫布、縮放重繪
│   ├── canvas_utils.py             #   繪圖函式：bbox / paste / prediction / tile overlay
│   ├── annotation_controller.py    #   Undo/Redo command stack（Add/Remove/Rename/Clear/Bulk）
│   ├── attribute_panel.py          #   右側屬性編輯面板（size_tag / crowded / difficulty / scene）
│   ├── class_mapping_service.py    #   classes.yaml 讀寫 + ClassCatalog 管理
│   ├── tile_panel.py               #   左下方 Tile view 面板（Size/Stride/Enable/Nav）
│   ├── constants.py                #   共用 UI 樣式常數
│   │
│   └── dialogs/                    #   對話框視窗
│       ├── __init__.py             #     匯出所有 dialog
│       ├── class_mapping_dialog.py #     Class mapping 編輯對話框
│       ├── error_analysis_dialog.py#     Error analysis 結果表格 + CSV 匯出
│       ├── statistics_dialog.py    #     Dataset statistics 四分頁表格 + JSON/CSV 匯出
│       ├── input_window.py         #     （legacy）物件名稱輸入
│       ├── showlab_window.py       #     Show Label 預覽視窗
│       ├── saveimg_window.py       #     Save Image 對話框
│       └── savelab_window.py       #     Save Label 格式選擇對話框
│
├── sdde/                           # 資料模型與服務層（純 Python，不依賴 PyQt6）
│   ├── __init__.py                 #   統一匯出所有 public API
│   ├── models.py                   #   核心資料模型：HBBBoxPx / HBBBoxYoloNorm / HBBAnnotation / ImageAnnotation
│   ├── config.py                   #   YAML name list 讀取
│   ├── class_catalog.py            #   ClassCatalog / ClassInfo — 類別目錄管理
│   ├── classes_yaml.py             #   classes.yaml 格式讀寫
│   ├── import_export.py            #   YOLO HBB parse / export、bbox txt export
│   ├── metadata_export.py          #   per-image annotation metadata → JSON / CSV
│   ├── attributes.py               #   研究屬性欄位定義、size_tag 計算（COCO 閾值）
│   ├── prediction.py               #   PredictionRecord、YOLO prediction 解析（含 confidence）
│   ├── error_analysis.py           #   IoU 計算、GT-vs-pred greedy matching、ErrorCase、CSV 匯出
│   ├── statistics.py               #   dataset 統計分析（class / size / bbox distributions）、JSON/CSV
│   ├── tile.py                     #   TileConfig / TileRect / compute_tile_grid / 座標轉換
│   ├── augmentation.py             #   PasteRecord（copy-paste transform 紀錄）、JSON/CSV 匯出
│   ├── project_config.py           #   ProjectConfig dataclass、YAML load / save
│   └── autosave.py                 #   autosave write / read / remove（JSON sidecar）
│
└── tests/                          # 單元測試（pytest）
    ├── test_models.py              #   HBBBoxPx / HBBAnnotation 測試
    ├── test_import_export.py       #   YOLO parse / export round-trip 測試
    ├── test_classes_yaml.py        #   classes.yaml 讀寫 round-trip
    ├── test_annotation_controller.py#  Undo / Redo command stack 測試
    ├── test_attributes.py          #   size_tag 計算、default_attributes 測試
    ├── test_metadata_export.py     #   build_annotation_records + JSON/CSV 匯出
    ├── test_prediction.py          #   YOLO prediction 解析（含/不含 confidence）
    ├── test_error_analysis.py      #   IoU / matching / error_type / CSV 匯出
    ├── test_statistics.py          #   dataset stats 計算 + JSON/CSV 匯出
    ├── test_tile.py                #   tile grid 計算 / 座標轉換 / annotations_in_tile
    ├── test_augmentation.py        #   PasteRecord / JSON/CSV 匯出
    ├── test_project_config.py      #   ProjectConfig YAML round-trip
    └── test_autosave.py            #   autosave write / read / remove
```

---

## 執行測試

```bash
pip install pytest
python -m pytest tests/ -q
```

目前共 54 個測試，涵蓋所有 `sdde/` 模組的核心邏輯。

---

## 架構設計原則

1. **資料與 UI 分離** — `sdde/` 是純 Python 資料層，不依賴 PyQt6；`gui/` 是 UI 層，呼叫 `sdde/` 的 API
2. **Command pattern** — 所有標註操作透過 `annotation_controller.py` 的 command stack，支援 Undo/Redo
3. **Tile 是視角不是資料** — tile 模式只影響畫布顯示，所有 annotation 始終寫回全圖座標
4. **Prediction 是獨立資料層** — prediction 疊在 GT 之上顯示，Accept 才轉為正式標註
5. **Autosave 是 sidecar** — 存在 `.autosave/` 目錄下的 JSON，不干擾原始 label 檔

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

第 6 欄為信心分數（可省略，預設 1.0）

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
