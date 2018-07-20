from setuptools import setup

setup(
    name='libnirs',
    version='0.0.1',
    description='Library for NIRS analysis',
    author='J Tatz',
    author_email='tatz.j@hsuky.neu.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['libnirs'],
    install_requires=[
        'numpy',
        'scipy',
        'numba'
    ],
    package_data={
        'libnirs': ['extc.npz'],
    }
)
