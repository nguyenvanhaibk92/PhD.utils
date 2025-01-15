## Why we want to do this?
- make/control your own working environment on whenever/wherever you will work on later. For example, each place has their own version of libraries, you have to deal with the version compatibility. Or in short-term, TACC updates their hardware/software, you have to deal with the new version of libraries.
- make your work reproducible. For example, what if you cannot reproduce your work after 1 year? how can you compare to the new methods?
- Make yourself cooler! :D 

## How to pull/run a docker image on TACC via Apptainer
1. request compute node 

```
idev -A <YOUR_RESOURCE_NAME> -p <NODE_TYPE> -m <NUMBER_of_MINUTES>
```

2. load the apptainer module

```
module load apptainer
```
you can do ``` module save ``` so you do not need to load apptainer every time you log in

3. pull the docker image

- for mixed JAX + TORCH versions.
```
apptainer pull docker://nguyenvanhaibk92/jaxtorch:v0.4.26
```

- for JaxAIStack (JAX + FLAX + Optax + Orbax + latex) version.
```
apptainer pull docker://nguyenvanhaibk92/jaxaistack_latex:2024.12.10
```
4. run the docker image interactively

```
apptainer shell --nv <LINK_TO_YOUR_IMAGE>.sif
```

5. Enjoy! Your terminal now is inside the docker image. check it our

```
python
```
you should see something like this
```
Python 3.11.11 (main, Dec  4 2024, 08:55:08) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Then, you can import jax and check the available devices
```
import jax
jax.devices()
```
You should see the cuda devices available on the compute node (if you have request GPU nodes).

---

## Visual Studio Code proxy jump (Thanks Wesley for help)

1. Install Visual Studio Code on your local machine
2. Install Remote - SSH extension
3. Open the command palette (Ctrl+Shift+P) and type "Remote-SSH: Open Configuration File". Access to ~/.ssh/config file
4. Add the following lines to the config file

```json
Host ls6
  HostName ls6.tacc.utexas.edu
  User <YOUR_TACC_ACCOUNT_NAME>

Host ls6-jax-ai-stack
  HostName <YOUR_SUCCESSFUL_REQUESTED_NODE> # for example c318-004
  RemoteCommand /opt/apps/tacc-apptainer/1.3.3/bin/apptainer shell --nv <LINK_TO_YOUR_IMAGE>.sif
  ForwardAgent yes
  ProxyJump ls6
  User <YOUR_TACC_ACCOUNT_NAME>
  RequestTTY yes
```
NOTE: 
- you might need to check `/opt/apps/tacc-apptainer/1.3.3/bin/apptainer` path since TACC might update in the future.
- you need to select "Enable Remote Command" in the Remote-SSH extension settings in Visual Studio Code.

5. Open the command palette (Ctrl+Shift+P) and type "Remote-SSH: Connect to Host..." and choose "ls6-jax-ai-stack"

6. You might need to enter TACC Token quickly to authenticate the connection, otherwise the connection will be closed.

7. Enjoy! Navigate to you project folder and start coding!

---
<!-- #### You can build/rebuild your own docker image depends on what you want to install further. I provided several Dockerfiles in the repository.  -->

