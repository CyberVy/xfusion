from setuptools import setup, find_packages

setup(
    name='xfusion',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "requests",
        'peft',
        'torch',
        'gradio==5.17.0',
        'diffusers>=0.32.2',
        'bitsandbytes>=0.45.3',
        'transformers>=4.48.3',
        'compel',
        'opencv-python',
        'tqdm',
    ],
    author='Eramth Ru',
    author_email='contanct@xsolutiontech.com',
    description='Xfusion',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://xsolutiontech.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
