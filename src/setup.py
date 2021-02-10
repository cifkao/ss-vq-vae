# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

import setuptools

setuptools.setup(
    name='ss-vq-vae',
    author='Ondřej Cífka',
    url='https://github.com/cifkao/ss-vq-vae',
    packages=setuptools.find_packages(),
    install_requires=[
        'bidict',
        'confugue>=0.1,<1',
        'librosa>=0.8,<1',
        'matplotlib',
        'numpy',
        'scikit_learn',
        'SoundFile',
        'tensorflow>=2,<3',
        'torch>=1.5,<2',
    ],
    python_requires='>=3.6',
)
