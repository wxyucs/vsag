from setuptools import setup, Extension
import glob

import platform


source_files = glob.glob('src/*.cpp')
source_files.append('python_bindings/binding.cpp')

compile_flags = [
    '-std=c++17',
    '-g'
]
if platform.processor() == "x86_64":
    from cpuinfo import get_cpu_info
    info = get_cpu_info()
    if 'flags' in info:
        if 'avx' in info['flags']:
            compile_flags.extend([
                '-mavx',
            ])
        if 'avx2' in info['flags']:
            compile_flags.extend([
                '-mavx2',
            ])
        if 'avx512f' in info['flags']:
            compile_flags.extend([
                '-mavx512f'
            ])
        if 'avx512vl' in info['flags']:
            compile_flags.extend([
                '-mavx512vl',
            ])
        if 'avx512bw' in info['flags']:
            compile_flags.extend([
                '-mavx512bw',
            ])
        if 'avx512dq' in info['flags']:
            compile_flags.extend([
                '-mavx512dq',
            ])
        if 'sse' in info['flags']:
            compile_flags.extend([
                '-msse',
            ])
        if 'avx' in info['flags'] and 'sse' in info['flags']:
            compile_flags.extend([
                '-Ofast'
            ])

ext_modules = [
    Extension(
        'vsag',
        sources=source_files,
        include_dirs=[
            'extern/hnswlib',
            'extern/json/single_include',
            'extern/spdlog/include',
            'include',
            'extern/pybind11/include',
            'extern/DiskANN/include',
        ],
        library_dirs=[
            '/opt/intel/lib/intel64_lin/',
            '/opt/intel/mkl/lib/intel64/',
            'extern/DiskANN/src/'
        ],
        libraries=["mkl_def", "mkl_core", "mkl_intel_ilp64", "mkl_intel_thread", "iomp5", "pthread", "m", "dl"],
        extra_objects=['/tbase-project/cluster/vsag/build/extern/DiskANN/src/libdiskann.a'],
        language='c++',
        extra_compile_args=compile_flags,
    )
]
setup(
    name='vsag',
    version='0.5.0',
    author='Tbase',
    author_email='tbase@antgroup.com',
    description='VSAG Package',
    ext_modules=ext_modules,
    install_requires=['pybind11', 'spdlog', 'numpy', 'h5py', 'oss2', 'scipy'],
)
