from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'This package is used to help with the interpretation and adversarial generation of pytorch models.'
LONG_DESCRIPTION = 'The "Adversarial Observation" framework is introduced to address concerns about the fairness and social impact of neural network models by allowing users to attack the network for adversarial resistance. This framework increases the explainability and transparency of the network using an adversarial swarm optimizer, making it more interpretable for stakeholders. The framework has the potential to improve the social impact of neural network models and enhance their effectiveness and efficiency by providing a user-friendly approach to adversarial testing and explainability.'

setup(
    name="Adversarial-Observation",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=["Jamil Gafur", "Study Duck", "Suely Oliveira", "Olivia Lang", "Justin Cha", "William Lai"],
    author_email="Jamil-gafur@uiowa.edu",
    license='MIT',
    packages=find_packages(),
    install_requires=["torch", "numpy", "tqdm", "pandas", "matplotlib", "sklearn", "os", "torchvision" ],
    keywords=['Interpretability', 'Adversarial', 'Explainability', 'Swarm', 'PSO', 'Pytorch', 'Neural Networks', 'Machine Learning', 'Deep Learning'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
