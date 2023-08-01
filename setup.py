from setuptools import setup, Extension
import glob
# 设置扩展模块


source_files = glob.glob('src/*.cpp')
source_files.append('python/binding.cpp')

ext_modules = [
    Extension(
        'vsag',  # 模块名
        sources=source_files,  # C++源文件
        include_dirs=[
            'extern/hnswlib',
            'extern/json/single_include',
            'extern/spdlog/include',
            'include',
            'extern/pybind11/include',
        ],  # 头文件目录
        language='c++',  # 编程语言
        extra_compile_args=[
            '-std=c++17',
            '-g',
            '-Ofast',
            '-mavx512f',  # 针对x86_64平台的编译选项
            '-mavx512dq',
            '-mavx512bw',
            '-mavx512vl',
            '-mavx',
            '-msse',
        ],
        extra_link_args=['-lgomp'],  # 额外的链接选项
    )
]

# 设置setup选项
setup(
    name='vsag',
    version='0.4.0',
    author='Tbase',
    author_email='tbase@antgroup.com',
    description='VSAG Package',
    ext_modules=ext_modules,
    install_requires=['pybind11', 'spdlog', 'numpy'],
)