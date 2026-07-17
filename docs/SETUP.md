# System Setup Guide — Linux Mint

This covers two independent things:

1. **The Python/TensorFlow stack** you need right now to run the LSTM +
   HDRL pipeline in `src/` against the probabilistic environment.
2. **Mininet-WiFi + Ryu**, which you do *not* need yet — the current
   `environment.py` is a mathematical simulation, not a real SDN emulation
   — but which the project will eventually need for the real-topology
   validation pass mentioned in the paper's proof-of-concept (Section VII).
   Installed and documented now so it's ready when you get there.

Tested against Linux Mint 21.x/22.x (Ubuntu 22.04/24.04 base). Commands
assume `bash`.

---

## Part 1 — Python / TensorFlow stack

### 1.1 System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl \
    python3-dev python3-pip python3-venv python3-tk \
    libhdf5-dev pkg-config
```

`libhdf5-dev` avoids a wheel-build failure some `h5py`/Keras-checkpoint
paths hit on Mint if only the pip wheel is attempted without the system
HDF5 headers present.

### 1.2 Virtual environment

Keep this isolated from any system Python (Mint ships Python for its own
tools; don't `pip install` into it directly).

```bash
python3 -m venv ~/venvs/cm_mtd
source ~/venvs/cm_mtd/bin/activate
pip install --upgrade pip setuptools wheel
```

### 1.3 Install project dependencies

```bash
cd cm_mtd/
pip install -r requirements.txt
```

**Why pure TensorFlow, no PyTorch:** an earlier iteration of this project
mixed TensorFlow (LSTM/DQN) with PyTorch (PPO). On a shared GPU the two
runtimes' allocators fought over device memory, and a broken `triton`
wheel bundled transitively with `torch` caused a hard segfault on import
on some driver/toolkit combinations. This version implements all three
learned components (LSTM, DQN, PPO) in TensorFlow/Keras, which sidesteps
both problems entirely. If you ever reintroduce PyTorch, be aware of that
history — `pip uninstall triton` was the previous workaround if you hit a
segfault on `import torch`.

### 1.4 GPU support (optional)

CPU is sufficient for the shrunk smoke-test config; a real 10k-episode
run benefits a lot from a GPU.

```bash
# Easiest path on Mint/Ubuntu: let pip pull the matching CUDA/cuDNN
pip install "tensorflow[and-cuda]"
```

Verify:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Pitfalls specific to Mint:

- Use Mint's **Driver Manager** GUI (or `ubuntu-drivers autoinstall`) to
  install the proprietary NVIDIA driver rather than manually downloading
  `.run` installers from NVIDIA — mixing the two is the single most common
  cause of a GPU not being detected after reboot.
- Don't also `apt install nvidia-cuda-toolkit`. The `tensorflow[and-cuda]`
  pip extra vendors its own CUDA/cuDNN; a second, differently-versioned
  toolkit from apt on the `LD_LIBRARY_PATH` causes TensorFlow to either
  silently fall back to CPU or crash with a cuDNN version-mismatch error.
- If you do put a model on GPU, `src/config_parser.configure_device()`
  already sets `memory_growth = True` on every visible GPU — this is what
  prevents TensorFlow from grabbing the entire GPU's memory up front,
  which matters if you ever run something else on the same card.

### 1.5 Sanity check

```bash
python3 scripts/generate_dummy_data.py --config config/config.yaml
python3 -m src.main --config config/config.yaml --mode train_lstm
```

You should see LSTM training epochs print and a fidelity number at the
end. This confirms the environment is wired correctly before you point
`config.yaml` at the real CICIDS-2017 `.npy` files from
`cicids2017.ipynb`.

---

## Part 2 — Mininet-WiFi + Ryu (for future real-topology validation)

Not required to run anything in `src/` today. Install this when you're
ready to replace `environment.py`'s probabilistic model with a real
emulated network, per the paper's Mininet-WiFi 2.5 / Ryu 4.34 setup
(Section VII).

### 2.1 Dependencies

```bash
sudo apt install -y git build-essential python3-dev python3-pip \
    hostapd wireless-tools iw wpasupplicant \
    net-tools iproute2 openvswitch-switch openvswitch-common
```

### 2.2 Mininet-WiFi from source

```bash
cd ~
git clone https://github.com/intrig-unicamp/mininet-wifi
cd mininet-wifi
sudo util/install.sh -Wlnfv
```

Flags: `-W` wireless tools, `-l` limit dependencies to what's needed,
`-n` mininet core, `-f` OpenFlow reference switch, `-v` Open vSwitch.

**Known pitfalls on Mint/newer Ubuntu bases:**

- The installer script has hardcoded `apt` package names that occasionally
  lag current Ubuntu releases (e.g. renamed `wpasupplicant` vs
  `wpa_supplicant` packages, or an `openvswitch-switch` version newer than
  what the script's OVS build step expects). If `install.sh` fails on a
  specific package, comment out that step and `apt install` the
  Mint-current equivalent manually, then re-run.
- **Run this in a VM or a dedicated machine, not your daily-driver
  laptop.** Mininet-WiFi creates and tears down real wireless/virtual
  network interfaces and can conflict with NetworkManager, sometimes
  dropping your actual Wi-Fi connection. `sudo service network-manager
  stop` before running experiments if you're on bare metal.
- If Open vSwitch was already installed via apt before running
  `install.sh`, you can end up with two conflicting OVS versions. Check
  with `sudo ovs-vsctl --version` and `dpkg -l | grep openvswitch`; if in
  doubt, `sudo apt purge openvswitch-switch openvswitch-common` before
  running the installer so it builds its own expected version.

Verify:

```bash
sudo mn --wifi --test pingall
```

### 2.3 Ryu SDN controller

The paper uses Ryu 4.34, but **upstream Ryu has been unmaintained since
2021** and its pinned `eventlet`/`six` dependencies frequently fail to
build against Python 3.10+ (which Mint 21/22 ship by default). Two
options:

**Option A — matches the paper exactly, needs an older Python:**

```bash
# Use pyenv or deadsnakes PPA to get Python 3.8 alongside your system Python
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev

python3.8 -m venv ~/venvs/ryu
source ~/venvs/ryu/bin/activate
pip install "eventlet==0.30.2" ryu==4.34
```

**Option B — recommended: `os-ken`, the actively-maintained OpenStack
fork of Ryu with the same API surface:**

```bash
pip install os-ken
```

`os-ken` is a drop-in replacement for the vast majority of Ryu
applications (same `ryu.app`-style controller API under `os_ken.*`) and
avoids the Python-version wall entirely. Prefer this unless you need
Ryu-specific behavior the paper's authors may have relied on.

Verify (Option B):

```bash
python3 -c "import os_ken; print(os_ken.__version__)"
```

### 2.4 Wiring it up later

When you're ready to replace the probabilistic `DigitalTwinNetworkEnv`
with a real Mininet-WiFi topology driven by an `os-ken`/Ryu controller,
the natural integration point is `environment.py`'s `step()` method: swap
the deterministic scan/compromise model for real OpenFlow flow-table
queries and actual HAM/RM mutation commands issued to the controller,
while keeping the same Gymnasium `step(action) -> (obs, reward, ...)`
contract so `models.py`'s DQN/PPO agents don't need to change at all.
