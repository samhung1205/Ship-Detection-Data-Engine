# Manual Smoke Test Checklist

這份清單是給重構前後使用的手動回歸基線。建議第一次實際操作專案時完整跑一輪，之後每次修改只挑受影響區塊回歸；如果有碰到 `gui/main_window.py`、匯入匯出、類別對照、autosave 或 prediction 流程，建議再跑完整輪。若修改開始碰到主視窗的 state 同步，請一併參考 `docs/state_refactor_inventory.md`；若修改開始碰到 copy-paste augmentation，請一併參考 `docs/paste_refactor_inventory.md`。

## 0. 先跑自動測試

```bash
pip install pytest
pytest -q
```

目前 repository 內建測試共 101 個，這一步應先全綠，再開始手動驗證。

## 1. 建議準備的測試素材

- 1 張可正常開啟的影像檔：`jpg`、`png`、`bmp` 任一種
- 1 份對應影像的 YOLO HBB label：格式為 `class_id xc yc w h`
- 1 份 prediction txt：格式為 `class_id xc yc w h [conf]`
- 1 張帶 alpha 的透明 PNG，用於 copy-paste

如果手邊還沒有完整素材，至少先準備一張影像與一張透明 PNG。

## 2. 啟動與初始狀態

- 啟動程式：`python GUI.py`
- 確認主視窗可正常開啟，沒有立即報錯
- 點開 `File -> Class mapping`
- 確認主路徑會讀取 `classes.yaml` 內容
- 確認 `data.yaml` 不屬於日常 class mapping 操作主流程
- 關閉對話框後，確認主視窗仍可正常操作

預期結果：

- 視窗標題顯示 `Ship Detection Data Engine`
- 類別對照視窗可正常讀取與關閉
- 沒有因為 `classes.yaml` 缺失或格式問題而崩潰

## 3. 開圖與縮放

- 點 `Open Image`
- 選一張測試影像
- 驗證畫布顯示正常
- 點 `zoom_in` / `zoom_out`
- 拖曳縮放 slider

預期結果：

- 影像可正常顯示
- 畫面縮放後不應出現殘影或框線錯位
- 狀態文字會更新目前座標與影像尺寸

## 4. 建立標註與屬性

- 點 `Create RectBox`
- 在畫布上點兩下建立一個框
- 再試一次，第二下故意點在第一下的左上方
- 在彈窗中選一個 class
- 點選右側列表中的該框
- 確認畫布上會有藍色半透明高亮預覽
- 在畫布上拖曳該框內部，確認可移動位置
- 再拖曳其中一個角點，確認可放大或縮小
- 再畫一個框，但在 class 選擇彈窗按取消
- 修改 `size_tag`、`crowded`、`difficulty_tag`、`scene_tag`
- 點一次 `Auto size_tag`
- 勾選 `Hide Box` 再取消

預期結果：

- 新框會出現在畫布與 `Box Labels`
- 已選取框可透過拖曳框內部移動，或拖曳角點調整大小；調整後畫布與資料列表應保持同步
- 無效第二點不應新增框，畫面應回到正常 overlay
- 點列表切換不同框時，高亮預覽應跟著切換，清除選取後不應殘留舊高亮
- 取消 class 選擇後，畫面不應殘留臨時綠框或錯誤框線
- 屬性修改後不會丟失
- `Hide Box` 只影響顯示，不應刪掉資料

## 5. Undo / Redo

- 建立至少 2 個框
- 使用 `Ctrl+Z`
- 再使用 `Ctrl+Shift+Z`
- 右鍵列表項目做 `Rename`
- 再測一次 `Undo` / `Redo`

預期結果：

- 新增、重新命名都能被 undo/redo
- undo 後框數量、名稱、屬性應同步回退

## 6. Label 載入、儲存、重載

- 點 `Save Label`
- 分別輸出：
  - `YOLO(v5~10)`
  - `Bounding Boxes`
- 記錄目前框數與類別名稱
- 清空標註或重新開同一張圖
- 點 `Load Label` 載回 YOLO label

預期結果：

- 匯出的檔案可成功產生
- 載回後框數量與類別應與儲存前一致
- 不應出現座標明顯偏移

## 7. Show Label 預覽

- 在有標註與 paste 內容時點 `Show Label`
- 檢查預覽視窗的顯示
- 用下拉選單切換 `All` 與單一類別

預期結果：

- 預覽視窗可正常開啟
- 類別過濾後只顯示對應框

## 8. Prediction Overlay

- 點 `Load preds`
- 載入 prediction txt
- 勾選與取消 `Show preds`
- 選一筆 prediction，測 `Accept`
- 再選另一筆 prediction，測 `Reject`
- 最後測 `File -> Clear predictions`

預期結果：

- prediction 以橘色虛線顯示
- `Accept` 後 prediction 會消失，並轉成正式 GT 框
- `Reject` 後 prediction 會被移除
- `Clear predictions` 會清空預測列表

## 9. Error Analysis

- 在同時存在 GT 與 prediction 的情況下執行 `Analysis -> Run error analysis...`
- 檢查結果表格是否有資料
- 勾選 `Bookmark`
- 填寫 `Notes`
- 測試 `Export CSV...`

預期結果：

- 對話框可正常開啟
- 表格欄位包含 `Type / IoU / GT class / Pred class / Conf`
- 匯出的 CSV 含 bookmark 與 notes

## 10. Dataset Statistics 與 Metadata Export

- 執行 `Analysis -> Dataset statistics...`
- 依序檢查四個分頁
- 測試 `Export JSON...` 與 `Export CSV...`
- 再測 `File -> Export annotation metadata (JSON)...`
- 再測 `File -> Export annotation metadata (CSV)...`

預期結果：

- 統計視窗可正常開啟
- 匯出檔可成功產生
- metadata 應包含 class 與屬性欄位

## 11. Tile View

- 設定 `Size` 與 `Stride`
- 開啟 `Tile view`
- 用 `<` `>` 切換 tile
- 再關閉 `Tile view`

預期結果：

- 畫布會顯示綠色 tile 邊框與外圍暗化
- 切換 tile 只改變視角，不應改動原框資料

## 12. Copy-Paste 與 Paste Metadata

- 點 `Choose`
- 選一張透明 PNG
- 點 `Paste Image`
- 在畫布上選定放置位置
- 用 `Resize / Rotate / Brightness / Contrast / HorizontalFlip` 調整
- 點 `Add`
- 點一下 paste 列表項目，確認藍色高亮預覽能顯示
- 右鍵 paste 列表項目做一次 `Rename`
- 再測一次單筆刪除
- 最後測一次 `Delete all`
- 測試 `Export paste metadata (JSON)...` 與 `Export paste metadata (CSV)...`

預期結果：

- 貼圖可顯示在畫布上
- 會自動生成對應 bbox
- 點 `Add` 不應因 paste metadata 記錄而崩潰
- 點選 paste 列表項目時，應有藍色半透明高亮預覽
- `Rename` 後列表名稱與 metadata class 應一致
- 單筆刪除與 `Delete all` 後，貼圖 overlay 與 metadata 應一起清掉
- 匯出的 metadata 會記錄 asset 路徑與 transform 參數

## 13. Autosave 與 Recovery

- 開圖並建立至少一個框
- 等待 autosave 週期，或確認狀態列出現 `Autosaved`
- 直接關閉程式，不先手動 `Save Label`
- 重新啟動並再開同一張圖
- 選擇是否恢復 autosave

預期結果：

- `.autosave/` 內會產生 sidecar 檔
- 重開後會跳出恢復詢問
- 選 `Yes` 時可恢復框與屬性
- 手動 `Save Label` 成功後，autosave 應被清除

## 14. Project Config

- 點 `File -> Save project config...`
- 儲存一份 `project_config.yaml`
- 修改 tile size / stride
- 再用 `File -> Open project config...` 載回剛剛的設定

預期結果：

- 設定檔可正常儲存與載入
- tile 參數與 autosave 秒數會跟著恢復

## 15. 建議記錄格式

每次重構後，至少記錄下面四項：

- 測試日期
- 修改範圍
- 自動測試結果
- 手動 smoke test 通過項目與失敗項目

簡單範例：

```text
Date: 2026-03-25
Change: label import/export refactor
Automated: 54 passed
Manual:
- [x] Open image / zoom
- [x] Create RectBox / attributes
- [x] Save / Load label
- [ ] Autosave recovery (not verified in this round)
Notes:
- YOLO round-trip visually consistent
```
