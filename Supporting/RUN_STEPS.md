# Steps to Run the Application in VSCode

## Prerequisites
- VSCode installed
- Python installed (Python 3.7+)
- All dependencies installed

## Step-by-Step Instructions

### 1. Open the Project in VSCode
   - Open VSCode
   - Go to `File` → `Open Folder`
   - Navigate to: `E:\Hasmita Documents\Major Project\Project`
   - Click "Select Folder"

### 2. Verify Python Interpreter
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type: `Python: Select Interpreter`
   - Select: `C:\Users\DELL\AppData\Local\Python\pythoncore-3.14-64\python.exe`
   - Or choose the interpreter that has all packages installed

### 3. Install Dependencies (if not already installed)
   - Open the integrated terminal: `Ctrl+`` (backtick) or `View` → `Terminal`
   - Run the following command:
     ```
     python -m pip install -r requirements.txt
     ```

### 4. Create Required Directory (if needed)
   - The application saves models in a `model/` directory
   - If it doesn't exist, create it:
     ```
     mkdir model
     ```

### 5. Run the Application
   
   **Option A: Using VSCode Run Button**
   - Click the "Run" button (play icon) in the top-right corner of VSCode
   - Or press `F5` (may require a launch.json configuration)
   
   **Option B: Using Terminal (Recommended)**
   - Open the integrated terminal: `Ctrl+``
   - Type:
     ```
     python main.py
     ```
   - Press Enter

### 6. Using the Application
   Once the GUI window opens:
   1. Click **"Upload Dataset"** → Select your CSV dataset file
   2. Click **"Preprocess Data"** → Wait for preprocessing to complete
   3. Click **"Decision Tree Classifier"** → Train/evaluate Decision Tree model
   4. Click **"Ridge Classifier"** → Train/evaluate Ridge Classifier model
   5. Click **"Performance Graph"** → View comparison graph
   6. Click **"Exit"** → Close the application

### 7. Troubleshooting
   
   **If you see import errors:**
   - Make sure the Python interpreter is correctly selected
   - Reload VSCode window: `Ctrl+Shift+P` → `Developer: Reload Window`
   - Verify packages: `python -m pip list | findstr scikit-learn`
   
   **If the GUI doesn't appear:**
   - Check the terminal for error messages
   - Make sure tkinter is available (usually comes with Python)
   - On Linux, you may need: `sudo apt-get install python3-tk`

## Quick Run Command
```bash
python main.py
```



