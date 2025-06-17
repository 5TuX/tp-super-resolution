# Installation instructions

1. Make sure you have uv python package and project manager installed. Instructions can be found here: [text](https://github.com/astral-sh/uv)
2. Clone the current repository on your machine:
    ```
    cd ~/Desktop/
    git clone https://github.com/5TuX/tp-super-resolution.git
    cd tp-super-resolution/
    ```
3. Install project dependencies in a virtual environement with uv:
    ```
    uv sync
    ```
4. Run the script to check that GPU is available:
    ```
    uv run gpu_check.py
    ```
    If everything is working correctly, you should see an output like this:
    ```
        ###### CHECK PYTORCH ######
        PyTorch imported

        tensor([[0.6000, 0.5968, 0.2197],
                [0.1701, 0.7872, 0.1009],
                [0.8937, 0.1301, 0.9057],
                [0.5979, 0.8791, 0.6934]])

        CUDA available: True

        CUDA version: 12.6
        cuDNN version: 90501
        Number of GPUs available: 1
        -> NVIDIA GeForce RTX 4090
    ```

# Run the notebook

```
uv run jupyter-lab tp_super_resolution.ipynb
```
