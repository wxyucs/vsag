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
    from cpufeature import CPUFeature
    if CPUFeature['AVX']:
        compile_flags.extend([
            '-mavx',
        ])
    if CPUFeature['AVX2']:
        compile_flags.extend([
            '-mavx2',
        ])
    if CPUFeature['AVX512f']:
        compile_flags.extend([
            '-mavx512f'
        ])
    if CPUFeature['AVX512vl']:
        compile_flags.extend([
            '-mavx512vl',
        ])
    if CPUFeature['AVX512bw']:
        compile_flags.extend([
            '-mavx512bw',
        ])
    if CPUFeature['AVX512dq']:
        compile_flags.extend([
            '-mavx512dq',
        ])
    if CPUFeature['SSE']:
        compile_flags.extend([
            '-msse',
        ])
    if CPUFeature['AVX'] and CPUFeature['SSE']:
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
        ],
        language='c++',
        extra_compile_args=compile_flags,
    )
]

setup(
    name='vsag',
    version='0.4.0',
    author='Tbase',
    author_email='tbase@antgroup.com',
    description='VSAG Package',
    ext_modules=ext_modules,
    install_requires=['pybind11', 'spdlog', 'numpy'],
)