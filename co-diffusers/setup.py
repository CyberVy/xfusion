from setuptools import setup, find_packages

setup(
    name='co-diffusers',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests',
        'torch',
        'torchsde',
        'diffusers',
        'compel',
        'tqdm',
        'flask'
    ],
    author='Eramth Ru',
    author_email='contanct@xsolutiontech.com',
    description='Diffusers on Colab',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://xsolutiontech.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
