# Setup and Verification for NVIDIA TensorFlow Container on Ubuntu

This guide provides step-by-step instructions to set up and verify the `nvcr.io/nvidia/tensorflow:24.06-tf2-py3` Docker container on an Ubuntu desktop with an NVIDIA RTX GPU. The container is optimized for NVIDIA GPUs and includes TensorFlow 2.x, CUDA 12.2, cuDNN 8.9, and Python 3.10, compatible with NVIDIA driver version 535 or later.

## Prerequisites
- **Ubuntu**: 20.04 or 22.04 (other versions may work but are untested here).
- **NVIDIA RTX GPU**: Compatible with compute capability 7.0+ (e.g., RTX 2080, 3080, 3090, 4090).
- **Docker**: Version 19.03 or later.
- **NVIDIA Driver**: Version 535 or later (confirmed compatible with CUDA 12.2).
- **NVIDIA Container Toolkit**: For GPU support in Docker.
- **NGC Account**: Free account for pulling images from NVIDIA GPU Cloud (NGC).

## Step 1: Install and Verify NVIDIA Driver
1. **Check NVIDIA Driver Version**:
   ```bash
   nvidia-smi
   ```
   - Ensure the driver version (top right of output) is 535 or higher.
   - If lower (e.g., 525), update the driver.

2. **Update NVIDIA Driver (if needed)**:
   ```bash
   sudo apt update
   sudo apt install -y nvidia-driver-535
   sudo reboot
   ```
   - Alternatively, download the driver from [NVIDIA’s website](https://www.nvidia.com/Download/index.aspx) (e.g., `NVIDIA-Linux-x86_64-535.183.01.run`) and install manually:
     ```bash
     wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run
     sudo systemctl stop gdm  # or lightdm, depending on your display manager
     sudo sh NVIDIA-Linux-x86_64-535.183.01.run
     sudo reboot
     ```
   - Verify after reboot:
     ```bash
     nvidia-smi
     ```

## Step 2: Install Docker
1. **Install Docker**:
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   ```
2. **Add User to Docker Group** (to run Docker without sudo):
   ```bash
   sudo usermod -aG docker $USER
   ```
   - Log out and back in for the group change to take effect.

3. **Verify Docker Installation**:
   ```bash
   docker --version
   ```
   - Ensure version is 19.03 or later.

## Step 3: Install NVIDIA Container Toolkit
1. **Set Up the NVIDIA Container Toolkit Repository**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-container-toolkit/stable/deb/nvidia-container-toolkit.list | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update
   ```

2. **Install NVIDIA Container Toolkit**:
   ```bash
   sudo apt install -y nvidia-container-toolkit
   ```

3. **Verify Installation**:
   ```bash
   nvidia-ctk --version
   ```
   - Should output something like `NVIDIA Container Toolkit CLI, version 1.17.8`.

## Step 4: Configure Docker for NVIDIA Runtime
1. **Configure Docker to Use NVIDIA Runtime**:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **Verify Configuration**:
   - Check Docker’s configuration file:
     ```bash
     cat /etc/docker/daemon.json
     ```
     Expected output:
     ```json
     {
         "runtimes": {
             "nvidia": {
                 "path": "nvidia-container-runtime",
                 "runtimeArgs": []
             }
         },
         "default-runtime": "nvidia"
     }
     ```
   - Check default runtime:
     ```bash
     docker info --format '{{.DefaultRuntime}}'
     ```
     - Should output `nvidia`.

## Step 5: Pull and Run the NVIDIA TensorFlow Container
1. **Log in to NVIDIA GPU Cloud (NGC)**:
   - Create a free NGC account at [https://ngc.nvidia.com/](https://ngc.nvidia.com/).
   - Generate an API key in the NGC dashboard (under “API Key” settings).
   - Log in to Docker with NGC:
     ```bash
     docker login nvcr.io
     ```
     - Username: `$oauthtoken`
     - Password: Your NGC API key

2. **Pull the Container**:
   ```bash
   docker pull nvcr.io/nvidia/tensorflow:24.06-tf2-py3
   ```

3. **Run the Container**:
   ```bash
   docker run --gpus all -it --rm nvcr.io/nvidia/tensorflow:24.06-tf2-py3
   ```
   - `--gpus all`: Enables GPU access.
   - `-it`: Runs in interactive mode.
   - `--rm`: Removes the container after exit.

## Step 6: Verify GPU Support in the Container
1. **Check TensorFlow GPU Support**:
   Inside the container, run:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices("GPU"))
   ```
   - Expected output lists your RTX GPU (e.g., `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`).

2. **Test NVIDIA-SMI in the Container**:
   Inside the container, run:
   ```bash
   nvidia-smi
   ```
   - Should display your RTX GPU (e.g., RTX 3090) and driver details.

## Troubleshooting
- **Driver Error** (`unsatisfied condition: cuda>=12.2`):
  - Verify driver version with `nvidia-smi`. Ensure it’s 535 or higher.
  - If lower, update to 535 (see Step 1).
  - Alternatively, use an older container like `nvcr.io/nvidia/tensorflow:23.12-tf2-py3` (CUDA 12.0, driver 520+).
- **Docker Runtime Error**:
  - If `docker info --format '{{.DefaultRuntime}}'` outputs `runc`, re-run:
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```
- **Permission Issues**:
  - Ensure you’re in the Docker group (`groups | grep docker`).
  - Run `sudo usermod -aG docker $USER` and log out/in if needed.
- **NGC Login Issues**:
  - Verify your NGC API key and use `$oauthtoken` as the username.
- **Container Fails to Start**:
  - Check Docker logs: `sudo journalctl -u docker`.
  - Ensure NVIDIA Container Toolkit is installed (`dpkg -l | grep nvidia-container`).

## Additional Resources
- [NVIDIA Container Toolkit Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [NVIDIA TensorFlow Release Notes](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/)
- [TensorFlow Docker Guide](https://www.tensorflow.org/install/docker)