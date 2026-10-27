# VS Code and Conda Setup Guide

To run this notebook on Windows or macOS, follow this complete walkthrough to install Visual Studio Code, set up a Conda environment with the required scientific libraries (`numpy`, `scipy`, and `ipykernel`), and execute the notebook.

---

## 1. Install Visual Studio Code

### On Windows
1. Download the installer from the official [VS Code download page](https://code.visualstudio.com/Download) (choose **Windows User Installer**).
2. Run the downloaded `.exe` installer.
3. During setup, accept the license agreement and leave the default install location.
4. On the **Select Additional Tasks** screen, check:
   - Add "Open with Code" action to Windows Explorer file context menu
   - Add "Open with Code" action to Windows Explorer directory context menu
   - Add to PATH (requires shell restart)
5. Click **Install**, then click **Finish**.

### On macOS
1. Download the installer from the official [VS Code download page](https://code.visualstudio.com/Download) (choose **Apple Silicon** for M1/M2/M3/M4 Macs or **Intel Chip** for older Macs).
2. Open your `Downloads` folder, double-click the downloaded `.zip` file if it did not extract automatically, and drag `Visual Studio Code.app` into your **Applications** folder.
3. Launch VS Code from Applications or via Spotlight (`Cmd + Space`).
4. To enable the command-line shortcut:
   - Press `Cmd + Shift + P` inside VS Code to open the Command Palette.
   - Type `shell command` and select **Shell Command: Install 'code' command in PATH**.

---

## 2. Install Miniforge (Conda)

Miniforge is the recommended, lightweight Conda distribution configured with the community-standard `conda-forge` channel.

### On Windows
1. Download the **Miniforge3 Windows 64-bit installer** (`Miniforge3-Windows-x86_64.exe`) from the Miniforge GitHub Releases page.
2. Run the installer, select **Just Me**, and follow the prompts.
3. Once finished, open the **Miniforge Prompt** (or Anaconda Prompt) from your Windows Start Menu.

### On macOS
1. Open the **Terminal** app (`Cmd + Space` $\rightarrow$ type `Terminal`).
2. Run one of the following commands depending on your processor architecture:
   - **Apple Silicon (M1/M2/M3/M4):**
     ```bash
     curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
     bash Miniforge3-MacOSX-arm64.sh
     ```
   - **Intel Mac:**
     ```bash
     curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
     bash Miniforge3-MacOSX-x86_64.sh
     ```
3. Follow the on-screen prompts: press `Enter` to scroll through the license, type `yes` to accept, press `Enter` for the default path, and type `yes` when asked to initialize Conda.
4. Restart your Terminal window so the `conda` command becomes active.

---

## 3. Create the Conda Environment

Open your terminal (**Miniforge Prompt** on Windows, or standard **Terminal** on macOS) and run the following commands:

```bash
# 1. Create a dedicated environment named 'chem_math' with Python 3.11
conda create -n chem_math python=3.11 -y

# 2. Activate the new environment
conda activate chem_math

# 3. Install NumPy, SciPy, and the Jupyter kernel interface
conda install numpy scipy ipykernel -y
```

---

## 4. Install VS Code Extensions

1. Open **Visual Studio Code**.
2. Click the **Extensions** icon on the left Activity Bar (or press `Ctrl + Shift + X` on Windows, `Cmd + Shift + X` on macOS).
3. Search for and install the following extensions published by Microsoft:
   - **Python** (ID: `ms-python.python`)
   - **Jupyter** (ID: `ms-toolsai.jupyter`)

---

## 5. Open and Run the Notebook in VS Code

1. In VS Code, navigate to **File** $\rightarrow$ **Open File...** (or **Open Folder...**) and select your downloaded notebook file (e.g., `modern_quantum_chemistry_ch1_lab-v3.ipynb`).
2. Look at the top-right corner of the notebook interface and click **Select Kernel** (or click the current Python environment shown).
3. In the dropdown list that appears at the top:
   - Select **Python Environments...**
   - Choose the environment named **`chem_math`** (it will display a path ending in `.../envs/chem_math/bin/python` on Mac or `...\envs\chem_math\python.exe` on Windows).
4. Run the first code cell (`import numpy as np ...`) by clicking the **Play** button to the left of the cell or pressing `Shift + Enter`.
5. When the version number prints beneath the cell without errors, your environment is ready to work through the lab exercises.