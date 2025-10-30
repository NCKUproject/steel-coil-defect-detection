# Steel Coil Defect Detection

使用 PyTorch 建立 **MLP** 與 **VGG16** 兩種模型，用於鋼捲表面瑕疵分類。
此專案包含完整的資料前處理、訓練、評估流程與 pre-commit 檢查機制。

---

## 專案架構

```
steel-coil-defect-detection/
│
├── configs/
│   └── base.yaml
│
├── data/
│   ├── raw/           # 原始影像資料，每個類別一個資料夾
│   └── splits/        # split.py 產生的 train.csv / test.csv
│
├── outputs/
│   ├── models/        # 儲存最佳模型 (best_mlp.pt, best_vgg16.pt)
│   ├── metrics/       # 儲存分類報告與混淆矩陣
│   └── preview/       # sanity_check 預覽影像
│
├── scripts/
│   ├── train/train_mlp.py # MLP 模型訓練
│   ├── train/train_vgg.py # VGG16 模型訓練
│   ├── eval/eval_mlp.py # MLP 模型評估
│   └── eval/eval_vgg.py # VGG16 模型評估
│
└── src/
    ├── dataprep/     # 資料前處理
    ├── models/       # 定義模型
    ├── training/     # 損失函數/優化器
    └── utils/        # 工具
```

---

## 環境建置

### Clone 專案

```powershell
git clone https://github.com/NCKUproject/steel-coil-defect-detection.git
cd steel-coil-defect-detection
```

---

### 啟用 PowerShell 腳本執行權限（Windows）

PowerShell 預設禁止執行虛擬環境腳本。請執行：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

> 此設定為臨時有效，重開機後需重新執行。

---

### 建立虛擬環境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

---

### 安裝 PyTorch

- 若有 **NVIDIA GPU (CUDA 12.1)**：

  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- 若使用 **CPU**：

  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

驗證：

```powershell
python - << 'PY'
import torch, torchvision
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

---

### 安裝其他依賴

```powershell
pip install -r requirement.txt
```

---

### 初始化 pre-commit（程式碼風格檢查）

```powershell
pre-commit install
python -m pre_commit run --all-files
```

---

## 資料準備流程

### 將原始資料解壓縮至 `data/raw/`

```bash
# 放壓縮檔在 data\raw\Simple_chinastell_data.zip
Expand-Archive -Force data\raw\Simple_chinastell_data.zip data\raw
```

```
data/raw/
├── 1_hole/
├── 2_blackscratch_long/
├── 3_whitescratch/
└── ...（共10類）
```

---

### 產生訓練與測試集（只需一次）

```powershell
python -m src.dataprep.split
```

輸出：

```
data/splits/train.csv
data/splits/test.csv
```

---

### 檢查資料（可選）

```powershell
python -m src.dataprep.sanity_check
```

輸出：

```
outputs/preview/train_batch_preview.png
```

---

## 模型訓練與評估

### MLP 模型

1. 編輯 `configs/base.yaml`：

   ```yaml
   model:
     name: mlp
   data:
     image_size: 128
   ```

2. 執行：

   ```powershell
   python -m scripts.train.train_mlp
   python -m scripts.eval.eval_mlp
   ```

---

### VGG16 模型

1. 編輯 `configs/base.yaml`：

   ```yaml
   model:
     name: vgg16
   data:
     image_size: 224
   ```

2. 執行：

   ```powershell
   python -m scripts.train.train_vgg
   python -m scripts.eval.eval_vgg
   ```

---

## 結果輸出

| 類別 | 路徑 |
|------|------|
| 模型 | `outputs/models/best_mlp.pt`、`best_vgg16.pt` |
| 準確率曲線 | `outputs/metrics/mlp_accuracy_curve.png` |
| 混淆矩陣 | `outputs/metrics/mlp_confusion_matrix.png` |
| 分類報告 | `outputs/metrics/mlp_classification_report.csv` |

---

## 總安裝步驟

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
pre-commit install
python -m pre_commit run --all-files
Expand-Archive -Force data\raw\Simple_chinastell_data.zip data\raw
python -m src.dataprep.split
python -m scripts.train.train_mlp
python -m scripts.eval.eval_mlp
```

---

## pre-commit 檢查機制

- 檢查檔案格式：`pre-commit run --all-files`
- 推code

```bash
git add .
git commit -m "commit message"
git push
```

- 如果用vscode內建的source control，可以設定自動執行 pre-commit
- 建立`.vscode/settings.json`，輸入以下內容

```json
{
    "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
    "terminal.integrated.env.windows": {
        "VIRTUAL_ENV": "${workspaceFolder}\\.venv",
        "PATH": "${workspaceFolder}\\.venv\\Scripts;${env:PATH}"
    },
    "python.terminal.activateEnvironment": true,
    "git.env": {
        "PATH": "${workspaceFolder}\\.venv\\Scripts;${env:PATH}"
    }
}

```
