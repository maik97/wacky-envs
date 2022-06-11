from distutils.core import setup
from setuptools import find_packages

setup(
    name='wacky-envs',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Create custom reinforcement learning environments with wacky-rl.',
    author='Maik Schürmann',
    author_email='maik.schuermann97@gmail.com',
    url='https://github.com/maik97/wacky-rl',
    download_url='https://github.com/maik97/wacky-envs/archive/refs/tags/v0.0.1.tar.gz',
    keywords=['rl', 'reinforcement-learning', 'environments', 'envs', 'python'],
    install_requires=[
        'gym',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
    ],
)
