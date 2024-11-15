from setuptools import setup, find_packages

setup(
    name='co-diffusers',             # 包名
    version='0.1.0',                 # 版本号
    packages=find_packages(),        # 自动发现项目中的包
    install_requires=[               # 依赖项
        'requests',
        'torch',
        'diffusers',
        'tqdm',
        'flask'
    ],
    author='Eramth Ru',              # 作者姓名
    author_email='contanct@xsolutiontech.com',  # 作者邮箱
    description='Diffusers on Colab',  # 简短描述
    long_description=open('README.md').read(),          # 详细描述
    long_description_content_type='text/markdown',      # 描述类型
    url='https://xsolutiontech.com', # 项目主页
    classifiers=[                                      # 分类信息
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',                            # Python 版本要求
)
