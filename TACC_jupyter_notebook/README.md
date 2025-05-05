# Running Local Jupyter Notebook Browse with TACC resources

This guide explains how to run a Jupyter Notebook server on a TACC (Texas Advanced Computing Center) supercomputer and connect to it from your local machine.

## Step 1: Connect to the TACC Login Node

```bash
ssh <USER>@<SYSTEM>.tacc.utexas.edu
```

Replace:
- `<USER>` with your TACC username
- `<SYSTEM>` with the system name (ls6, vista, frontera, etc.)

## Step 2: Set Up Your Python Environment

It's recommended to create and use a Python environment for your Jupyter Notebook:

```bash
# If using Conda
module load anaconda3
conda create -n jupyter_env python=3.9 ipykernel jupyter
conda activate jupyter_env

# If using venv
module load python3
python3 -m venv jupyter_env
source jupyter_env/bin/activate
pip install jupyter ipykernel
```

## Step 3: Request a Compute Node

Request an interactive session on a compute node:

```bash
# For SLURM-based systems
idev -N 1 -n 1 -p normal -t 02:00:00  # Requests 1 node for 2 hours
```

After allocation, you'll be connected to a compute node with a name like `<NODE_ID>.<SYSTEM>.tacc.utexas.edu`.

## Step 4: Start the Jupyter Notebook Server

Navigate to your desired working directory and start the notebook server:

```bash
cd /path/to/your/project
jupyter notebook --no-browser --ip=0.0.0.0
```

You'll see output similar to:
```
[I xx:xx:xx.xxx NotebookApp] Jupyter Notebook is running at:
[I xx:xx:xx.xxx NotebookApp] http://<NODE_ID>.<SYSTEM>.tacc.utexas.edu:8888/?token=<TOKEN_SERIES>
```

Make note of:
- The compute node ID (`<NODE_ID>`) 
- The token series (`<TOKEN_SERIES>`)

## Step 5: Create SSH Tunnel from Your Local Machine

Open a new terminal window on your local machine and create an SSH tunnel:

```bash
ssh -N -L 8889:<NODE_ID>.<SYSTEM>.tacc.utexas.edu:8888 <USER>@<SYSTEM>.tacc.utexas.edu
```

This command will:
- Create a secure tunnel between your local port 8889 and the remote Jupyter server (port 8888)
- Route through the TACC login node to reach the compute node
- Appear to hang with no output (this is normal - leave it running)

## Step 6: Connect to Your Jupyter Notebook

Open a web browser on your local machine and go to:

```
http://localhost:8889/?token=<TOKEN_SERIES>
```

Use the token from Step 4 for authentication.

You now have a Jupyter Notebook server running on a TACC supercomputer, accessible from your local browser!

## Tips

- To create a convenient alias for the Jupyter command, add this to your `.bashrc` or `.bash_profile`:
  ```bash
  alias jupnode='jupyter notebook --no-browser --ip=0.0.0.0'
  ```
  
- If you're working with specific frameworks (TensorFlow, PyTorch, etc.), install them in your Python environment before launching Jupyter.

- Remember to close your Jupyter server and exit your compute node allocation when finished to free resources.
