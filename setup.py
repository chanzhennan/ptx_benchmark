import os
import glob
import multiprocessing
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.command.install import install
import shutil

name = 'gpumark'
current_directory = os.path.abspath(os.path.dirname(__file__))

sources = [
    os.path.join('gpumark', 'pybind.cpp'),
    os.path.join('gpumark', 'kernels', f'addpi.cu'),
]


class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        num_cores = multiprocessing.cpu_count()

        for ext in self.extensions:
            ext.extra_compile_args = ['-j', '-v', str(num_cores)]  # 使用-j选项设置线程数
        super().build_extensions()


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        print('Running custom install command...')


setup(
    name=name,
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name,
            sources=sources,
            library_dirs=['/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs'],  # 添加库路径
            extra_link_args=['-lcuda'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-fopenmp',
                    '-lgomp',
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--generate-line-info',
                    '-Xptxas=-v',
                    '-lcuda',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
        'install': CustomInstallCommand,
    },
)
