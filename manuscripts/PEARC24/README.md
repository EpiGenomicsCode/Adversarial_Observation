# Adversarial Robustness and Explainability of Machine Learning Models

This repository accompanies the PEARC'24 paper:

> Gafur J, Goddard S, Lai W. "Adversarial Robustness and Explainability of Machine Learning Models." In *Practice and Experience in Advanced Research Computing 2024: Human Powered Computing*, pp. 1–7, 2024.

## Introduction

Deep neural networks have achieved remarkable accuracy across a range of classification tasks, yet they remain vulnerable to adversarial examples—carefully crafted perturbations that are imperceptible to humans but cause confident misclassifications. Understanding *how* and *why* these attacks succeed is essential for deploying machine learning models in safety-critical domains such as autonomous driving, medical imaging, and cybersecurity.

This work investigates adversarial robustness through a **black-box attack framework** built on **Particle Swarm Optimization (PSO)**. Unlike gradient-based methods (e.g., FGSM, PGD) that require access to model internals, PSO treats the target classifier as an opaque function, making the approach applicable to any deployed model regardless of architecture. We pair the attack with a detailed **explainability pipeline** that tracks, for every particle at every iteration, the softmax confidence landscape, pixel-wise perturbation magnitude, and the trajectory through the search space. Together, these analyses reveal the structural weaknesses of a trained model and provide interpretable evidence of where decision boundaries are most fragile.

The framework is demonstrated on a **convolutional neural network (CNN) trained on MNIST**, chosen as a well-understood baseline that allows clear visualization of adversarial perturbations. The codebase is designed to be extensible to other datasets and model architectures.

### Key Contributions

- A **PSO-based black-box adversarial attack** that generates misclassified images without gradient access.
- An **iteration-level explainability pipeline** that logs confidence values, softmax outputs, and pixel-wise differences, providing a window into the attack dynamics.
- Reproducible analysis artifacts (images and structured JSON logs) that support further research into model robustness.

## Citing This Work

If you use this code in your research, please cite:

```bibtex
@incollection{gafur2024adversarial,
  title     = {Adversarial Robustness and Explainability of Machine Learning Models},
  author    = {Gafur, Jamil and Goddard, Steve and Lai, William},
  booktitle = {Practice and Experience in Advanced Research Computing 2024: Human Powered Computing},
  pages     = {1--7},
  year      = {2024}
}
```

---

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request. Ensure that commit messages are clear, tests are updated as needed, and code follows the existing conventions.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
