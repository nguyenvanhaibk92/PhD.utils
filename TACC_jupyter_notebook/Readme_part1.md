# Part 1: Running Jupyter Notebooks on TACC on Browser Interface

A step-by-step tutorial for students to run Jupyter Notebooks on TACC (Texas Advanced Computing Center) supercomputers and access them from your local machine.

---

## 📋 Prerequisites

Before you begin, make sure you have:

- ✅ A TACC account (create one at [https://accounts.tacc.utexas.edu/register](https://accounts.tacc.utexas.edu/register))
- ✅ SSH access configured on your local machine
- ✅ Basic familiarity with terminal/command line

---

## 🎯 Overview

This tutorial will guide you through:

1. Connecting to TACC
2. Requesting compute resources
3. Setting up your Python environment
4. Starting a Jupyter server
5. Creating an SSH tunnel
6. Accessing your notebook in a browser

**Total time:** ~10-15 minutes

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

# Install Jupyter
pip install jupyter ipykernel

# You can use pip to install all necessary packages for your project
# Here I use Jax to check GPU availability later
pip install jax[cuda12]
```

---

## Step 4: Start the Jupyter Notebook Server

1. **Navigate to your project directory:**

   ```bash
   cd /path/to/your/project
   ```

   Or create a new directory at workspace directory:

   ```bash
   mkdir -p $WORK/my_notebooks
   cd $WORK/my_notebooks
   ```

2. **Start Jupyter:**

   ```bash
   jupyter notebook --no-browser --ip=0.0.0.0
   ```

   **What these flags do:**

   - `--no-browser` → Don't try to open a browser (we're on a remote server)
   - `--ip=0.0.0.0` → Allow connections from outside the node

3. **Copy the connection information:**

   You'll see output like this:

   ```text
   [I 2025-11-18 10:19:20.875 ServerApp] Jupyter Server 2.17.0 is running at:
   [I 2025-11-18 10:19:20.875 ServerApp] http://c608-082.vista.tacc.utexas.edu:8888/tree?token=a7fe3990268c89ca397ef901184755f52834068683f6f77d
   ```

   **📝 Write down:**

   - **Node ID:** `c608-082` (from the URL)
   - **Token:** `a7fe3990268c89ca397ef901184755f52834068683f6f77d` (the long string after `token=`)

   **⚠️ Keep this terminal window open!** The Jupyter server is running here.

---

## Step 5: Create SSH Tunnel from Your Local Machine

Now you'll create a secure connection from your local computer to the Jupyter server.

1. **Open a NEW terminal window** on your local machine (keep the TACC terminal open)

2. **Create the SSH tunnel:**

   ```bash
   ssh -N -L 8889:<NODE_ID>.<SYSTEM>.tacc.utexas.edu:8888 <USER>@<SYSTEM>.tacc.utexas.edu
   ```

   **Replace:**

   - `<NODE_ID>` → The compute node ID from Step 4 (e.g., `c608-082`)
   - `<SYSTEM>` → The TACC system name (e.g., `vista`)
   - `<USER>` → Your TACC username

   **Example:**

   ```bash
   ssh -N -L 8889:c608-082.vista.tacc.utexas.edu:8888 nvhai@vista.tacc.utexas.edu
   ```

   ![Example: Jupyter connection info](images/Screenshot%202025-11-18%20at%2010.24.31%E2%80%AFAM.png)

3. **What happens:**

   - The command will prompt for your TACC password
   - After entering it, the terminal will appear to "hang" (no output)
   - **This is normal!** The tunnel is running. Leave this terminal open.

   **What this does:** Creates a secure tunnel that forwards your local port 8889 to the Jupyter server on the compute node.

---

## Step 6: Access Your Jupyter Notebook

1. **Open your web browser** (Chrome, Firefox, Safari, etc.)

2. **Go to:**

   ```text
   http://localhost:8889/?token=<TOKEN>
   ```

   Replace `<TOKEN>` with the token you copied in Step 4.

   **Example:**

   ```text
   http://localhost:8889/?token=a7fe3990268c89ca397ef901184755f52834068683f6f77d
   ```

   ![Screenshot: Jupyter Notebook interface](images/Screenshot%202025-11-18%20at%2010.27.17%E2%80%AFAM.png)

3. **You should see the Jupyter Notebook interface!** 🎉

   - Browse your files
   - Create new notebooks
   - Open existing notebooks
   - Run your code on TACC's powerful compute nodes

   **Example:**
   Let create a notebook with the name "test.ipynb", then check GPU availability with Jax.

   ![Screenshot: Example notebook running JAX on GPU](images/Screenshot%202025-11-18%20at%2010.30.21%E2%80%AFAM.png)

---
