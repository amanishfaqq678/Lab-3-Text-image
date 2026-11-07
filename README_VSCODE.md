# Lab 03 — VS Code Ready Project

## 1. Requirements
- Windows 10/11
- Python 3.10 or newer
- Visual Studio Code
- VS Code Python extension

## 2. Open the project
1. Extract `Lab03_VSCode_Ready.zip`.
2. Open VS Code.
3. Select **File → Open Folder**.
4. Select the extracted `Lab03_VSCode_Ready` folder.
5. Open `main.py`.

## 3. Create a virtual environment (VS Code terminal)
Open **Terminal → New Terminal**, then run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt terminal and run:
```cmd
.venv\Scripts\activate.bat
```

## 4. Install libraries
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Select Python interpreter
Press `Ctrl+Shift+P` → type `Python: Select Interpreter` → choose the interpreter inside `.venv`.

## 6. Run the complete lab
In the VS Code terminal:
```powershell
python main.py
```
Expected key results:
- Text feature matrix shape: `(3, 11)`
- RGB dummy image shape: `(100, 100, 3)` = 30,000 values
- Grayscale image shape: `(100, 100)` = 10,000 values
- Flattened image vector shape: `(10000,)`
- Resized 64x64 image vector shape: `(4096,)`
- Three differently sized images resized to 64x64 each return `(4096,)`

Pixel values are generated data, so the exact first ten values can differ if you change the dummy image generator.

## 7. Home assignment
Add 10 images to `data/class_a/` and 10 images to `data/class_b/`. Run `python main.py` again. The script saves `dataset.npz` with:
- `X`: image feature matrix
- `y`: class labels

The core lab does not require external image files. It creates its own 100x100 dummy image.

## 8. Common fixes
- `python is not recognized`: try `py --version`; use `py main.py`.
- `No module named ...`: activate `.venv`, then run `python -m pip install -r requirements.txt`.
- Wrong interpreter: reselect `.venv` using `Python: Select Interpreter`.
- Activation policy error: use Command Prompt and `.venv\Scripts\activate.bat`.

## Viva preparation
Read `VIVA_AND_ANSWERS.md`. Be able to explain BoW, vocabulary, feature matrix, grayscale, pixel, flattening, and why image resizing creates a fixed vector length.
