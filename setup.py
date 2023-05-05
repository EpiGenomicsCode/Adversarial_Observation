from setuptools import setup, find_packages

VERSION = '0.0.4'
DESCRIPTION = 'Adversarial-Observation: A package for adversarial generation and interpretation of PyTorch models'
LONG_DESCRIPTION = '''Adversarial-Observation is a Python package that provides a framework for improving the transparency, interpretability, and social impact of PyTorch models. The package includes tools for generating adversarial attacks against neural network models, which can help improve their adversarial robustness and reduce the potential for unintended consequences in real-world applications.

The package includes an adversarial swarm optimizer that allows users to explore the behavior of their models under various adversarial conditions. This optimizer can be used to improve the interpretability of the model by identifying key features and decision boundaries that are most vulnerable to adversarial attacks.

Adversarial-Observation is intended for developers and researchers working in the field of machine learning, particularly those focused on deep learning and neural network models. The package is designed to be easy to use and requires only a basic understanding of Python and PyTorch to get started.
'''

setup(
    name="Adversarial-Observation",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=["Jamil Gafur", "Study Duck", "Olivia Lang", "Justin Cha", "William Lai"],
    author_email="Jamil-gafur@uiowa.edu",
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
                        "imageio",
                        "matplotlib",
                        "numpy",
                        "torch",
                        "pandas"
                      ],
    examples_require=[ 
                        "torchvision",
                        "sklearn",
                        "tqdm",
                        "pickle",
                        "tqdm"
                        ],

    keywords=['Interpretability', 
              'Adversarial', 
              'Explainability', 
              'Swarm', 
              'PSO', 
              'Pytorch', 
              'Neural Networks', 
              'Machine Learning', 
              'Deep Learning'
              ],
    url="https://github.com/EpiGenomicsCode/Adversarial_Observation",
     project_urls={
        'Bug Tracker': 'https://github.com/EpiGenomicsCode/Adversarial_Observation/issues',
        'Documentation': 'https://example.com/documentation',
        'Source Code': 'https://github.com/EpiGenomicsCode/Adversarial_Observation',
        'PyPI Page': 'https://pypi.org/project/Adversarial-Observation/',
        'CI/CD Pipelines': 'https://example.com/ci-cd-pipelines'
    },
    classifiers= [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)
