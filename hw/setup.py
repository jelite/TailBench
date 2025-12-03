from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='coarse_cuda',
    ext_modules=[
        CUDAExtension(
            'coarse_cuda',
            ['coarse_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
32