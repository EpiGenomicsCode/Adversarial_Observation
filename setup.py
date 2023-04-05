from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A things package'
LONG_DESCRIPTION = 'A package that  does things'

setup(
    name="Adversarial_Observation",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="<Your Name>",
    author_email="<your email>",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='conversion',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
