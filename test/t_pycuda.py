import pycuda.driver as cuda
cuda.init()

device = cuda.Device(0)
attrs = device.get_attributes()

print("GPU:", device.name())
print("SM count:", attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT])
print("Max warps per SM:", attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR] // 32)
print("Max threads per SM:", attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR])
print("Max blocks per SM:", attrs[cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR])
print("Max registers per SM:", attrs[cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR])
print("Max shared mem per SM (bytes):", attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR])
