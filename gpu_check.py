# Two handy functions for testing GPU setup in PyTorch and Tensorflow


def gpu_check_pytorch():
    print("###### CHECK PYTORCH ######\n")
    import torch

    print("PyTorch imported\n")
    x = torch.rand(4, 3)
    print(x)
    cuda_test = torch.cuda.is_available()
    print("\nCUDA available:", cuda_test)
    if cuda_test:
        print("\nCUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
    print("Number of GPUs available:", c := torch.cuda.device_count())
    for i in range(c):
        print("->", torch.cuda.get_device_name(i))


def gpu_check_tensorflow():
    print("\n###### CHECK TENSORFLOW ######\n")
    import tensorflow as tf

    print("\nTensorflow imported\n")
    x = tf.random.uniform((4, 3))
    print(x)
    cuda_test = tf.test.is_built_with_cuda()
    print("\nBuilt with cuda:", cuda_test)
    if cuda_test:
        print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
        print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
    devices = tf.config.list_physical_devices("GPU")
    print("Number of GPUs Available:", len(devices))
    for device in devices:
        info = tf.config.experimental.get_device_details(device)
        print("->", info["device_name"])


gpu_check_pytorch()
# gpu_check_tensorflow()
