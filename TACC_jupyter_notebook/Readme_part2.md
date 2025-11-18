# Part 2: Running Jupyter Notebooks on TACC on Visual Studio Code (VSCode)

A step-by-step tutorial for students to run Jupyter Notebooks on TACC (Texas Advanced Computing Center) supercomputers using Visual Studio Code (VSCode) as your development environment.

---

## 📋 Prerequisites

Before you begin, make sure you have:

- ✅ A TACC account (create one at [https://accounts.tacc.utexas.edu/register](https://accounts.tacc.utexas.edu/register))
- ✅ Visual Studio Code installed on your local machine
- ✅ SSH access configured on your local machine
- ✅ Basic familiarity with terminal/command line

---

## 🎯 Overview

This tutorial will guide you through:

1. Connecting to TACC
2. Requesting compute resources
3. Setting up your Python environment
4. Configuring automatic Python environment activation
5. Installing and configuring VSCode with Remote - SSH extension
6. Setting up SSH configuration for login and compute nodes
7. Connecting to the compute node via VSCode
8. Running Jupyter notebooks in VSCode

---

## Step 0: Create Your TACC Account

If you haven't already, create a TACC account at:
👉 [https://accounts.tacc.utexas.edu/register](https://accounts.tacc.utexas.edu/register)

1. Wait for account approval (usually within 24-48 hours).
2. Ask PI/Advisor to add you to the resource.

---

## Step 1: Connect to the TACC Login Node

Open a terminal on your local machine and connect to TACC:

```bash
ssh <USER>@<SYSTEM>.tacc.utexas.edu
```

**Replace the placeholders:**

- `<USER>` → Your TACC username (e.g., `nvhai`)
- `<SYSTEM>` → The TACC system name (e.g., `ls6`, `vista`, `frontera`)

**Example:**

```bash
ssh nvhai@vista.tacc.utexas.edu
```

<img src="images/Screenshot 2025-11-18 at 9.54.53 AM.png" alt="TACC Login Screenshot" width="600"/>

---

## Step 2: Request a Compute Node

The login node is for setup only. You need a compute node to run Jupyter and set up your Python environment.

Request an interactive session:

```bash
idev -A DMS22021 -N 1 -n 1 -p gh-dev -t 02:00:00
```

**What this means:**

- `-A DMS22021` → The resource `DMS22021`, which PI/Advisor has added you to.
- `-N 1` → Request 1 node
- `-n 1` → Use 1 task
- `-p gh-dev` → the GPU node type. [Click this link for more details](https://docs.tacc.utexas.edu/hpc/vista/#:~:text=Table%204.-,Production%20Queues)
- `-t 02:00:00` → Reserve for 2 hours (adjust as needed). Depends on the GPU node type, the maximum time you can request is 48 hours.

**After running this command:**

- You'll wait briefly while TACC allocates resources
- Once allocated, you'll be automatically connected to a compute node
- Your prompt will change to show the node name (e.g., `c608-082.vista.tacc.utexas.edu`)

**⚠️ Important:** Note the compute node name! You'll need it in Step 6. For example, in the screenshot, the compute node name is `c608-082`.

<img src="images/Screenshot 2025-11-18 at 10.02.39 AM.png" alt="idev session screenshot" width="600"/>

---

## Step 3: Set Up Your Python Environment

Now that you're on a compute node, you'll set up your Python environment with Jupyter installed.

```bash
# Load Python module
module load python3

# Go to workspace directory (by default, you are in $HOME directory)
# Create a directory where you install all future environments
cd $WORK/
mkdir -p PYTHON_ENVs
cd PYTHON_ENVs

# Create a virtual environment
# (Set the python environment name that describes the best for your project)
# (here we set it to "jupyter_env" for this project)
python3 -m venv jupyter_env

# Activate the environment
source $WORK/PYTHON_ENVs/jupyter_env/bin/activate

# Install Jupyter (required for VSCode notebook support)
pip install jupyter ipykernel

# You can use pip to install all necessary packages for your project
# Here I use Jax to check GPU availability later
pip install jax[cuda12]
```

---

## Step 4: Configure Automatic Python Environment Activation

**⚠️ Important:** This step ensures your Python environment is automatically activated whenever you connect to a compute node.

Run the command to append the activation command to your `.bashrc` file:

```bash
echo 'source $WORK/PYTHON_ENVs/jupyter_env/bin/activate' >> ~/.bashrc
```

**💡 Tip:** If you create additional Python environments in the future, you can update this command or add multiple activation commands to your `.bashrc`.

---

## Step 5: Install and Configure Visual Studio Code

### 5.1: Install VSCode Command Line Tool

1. **Open Visual Studio Code.**
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac) to open the Command Palette.
3. Type and select: **Shell Command: Install 'code' command in PATH**.
4. Restart your terminal.

Now you can open VSCode from your terminal with:

```bash
code .
```

### 5.2: Install Remote - SSH Extension

On the VSCode interface, find the extension **"Remote - SSH"** and install it.

![Screenshot: VSCode Remote - SSH extension](images/Screenshot%202025-11-18%20at%2011.06.43%E2%80%AFAM.png)

---

## Step 6: Set Up SSH Configuration File

Run the command to open the SSH config file:

```bash
code ~/.ssh/config
```

Add the following configuration:

```ssh
Host <LOGIN_HOST_ALIAS>
    HostName <SYSTEM>.tacc.utexas.edu
    User <USER>

Host <COMPUTE_NODE_ALIAS>
    HostName <COMPUTE_NODE_HOSTNAME>
    ForwardAgent yes
    ProxyJump <LOGIN_HOST_ALIAS>
    User <USER>
    RequestTTY yes
```

**Replace the placeholders:**

- `<LOGIN_HOST_ALIAS>` → A friendly name for your login node (e.g., `vista`)
- `<SYSTEM>` → The TACC system name (e.g., `vista`, `ls6`, `frontera`)
- `<USER>` → Your TACC username
- `<COMPUTE_NODE_ALIAS>` → A friendly name for your compute node (e.g., `jupyter_env`)
- `<COMPUTE_NODE_HOSTNAME>` → The compute node name from Step 2 (e.g., `c608-082`)

**Example:**

```ssh
Host vista
    HostName vista.tacc.utexas.edu
    User nvhai

Host jupyter_env
    HostName c608-082
    ForwardAgent yes
    ProxyJump vista
    User nvhai
    RequestTTY yes
```

**⚠️ Important:** You have to update the `<COMPUTE_NODE_HOSTNAME>` every time you request a new compute node.

---

## Step 7: Connect to the Compute Node via VSCode

1. **In VSCode, click the bottom left orange icon** to open the SSH connection tool.

   ![Screenshot: VSCode SSH connection tool](images/Screenshot%202025-11-18%20at%2011.20.23%E2%80%AFAM.png)

2. **Select `<COMPUTE_NODE_ALIAS>`** (the alias you configured in Step 6, e.g., `jupyter_env`).

3. **Authenticate the connection:**
   - You'll be prompted to enter your TACC password
   - You may also need to enter your TACC token (2FA)

4. **Navigate to your project directory:**

   You'll be in the HOME directory by default. Open the terminal in VSCode and run:

   ```bash
   cd /path/to/your/project
   ```

   **Example:**

   ```bash
   cd $WORK/my_notebooks
   pwd
   ```

   **💡 Tip:** Press `Cmd` (Mac) or `Ctrl` (Windows/Linux) + left click on the path in the terminal output to navigate to that directory in the file browser.

   ![Screenshot: VSCode File Browser Example](images/Screenshot%202025-11-18%20at%2011.29.50%E2%80%AFAM.png)

5. **Verify you're in the project directory:**

   ![Screenshot: VSCode in Project Directory](images/Screenshot%202025-11-18%20at%2011.31.40%E2%80%AFAM.png)

6. **Select the Python interpreter:**

   Click on the Python version in the bottom right corner of VSCode and select the interpreter from your virtual environment:

   ```text
   $WORK/PYTHON_ENVs/jupyter_env/bin/python
   ```

   ![Screenshot: VSCode Select Python Interpreter](images/Screenshot%202025-11-18%20at%2011.33.26%E2%80%AFAM.png)

7. **You're all set!** 🎉 Now you can start coding and running Jupyter notebooks in VSCode!

   ![Screenshot: VSCode Notebook Output Example](images/Screenshot%202025-11-18%20at%2011.36.30%E2%80%AFAM.png)

---
