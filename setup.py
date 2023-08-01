from setuptools import setup, Extension
import glob


source_files = glob.glob('src/*.cpp')
source_files.append('python/binding.cpp')

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
        extra_compile_args=[
            '-std=c++17',
            '-g',
            '-Ofast',
            '-mavx512f',
            '-mavx512dq',
            '-mavx512bw',
            '-mavx512vl',
            '-mavx',
            '-msse',
        ],
        extra_link_args=['-lgomp'],
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