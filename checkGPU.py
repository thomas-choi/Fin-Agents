import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("Physical Devices:", tf.config.list_physical_devices('GPU'))
print("GPU Device Name:", tf.test.gpu_device_name())