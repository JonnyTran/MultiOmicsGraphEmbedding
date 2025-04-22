# MultiOmicsGraphEmbedding

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MultiOmicsGraphEmbedding is a PyTorch-based package designed for representation learning on heterogeneous networks. This repository is aimed at facilitating experimentation with various Graph Neural Network (GNN) models, metrics, and datasets, leveraging powerful libraries such as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and [DGL](https://www.dgl.ai/). Whether you're a researcher or a practitioner, this package helps you explore and apply graph representation learning techniques efficiently.

## Features

- **Modular Architecture**: Experiment with different GNN architectures, data loaders and metrics effortlessly.
- **Heterogeneous Network Support**: Handle multi-relational and multi-modal data seamlessly.
- **Integration**: Compatible with PyTorch Geometric, DGL, and NetworkX for flexibility.
- **Reproducibility**: Easy-to-use pipeline for reproducible experiments.

## Installation

Install the package directly from source with

```bash
pip install git+https://github.com/JonnyTran/MultiOmicsGraphEmbedding
```

## Install from source

### Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- DGL
- Other dependencies listed in `requirements.txt`

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/JonnyTran/MultiOmicsGraphEmbedding.git
cd MultiOmicsGraphEmbedding
```

### 2. Run the Example

Run a sample script to test the setup and see the package in action:

```bash
python examples/sample_script.py
```

### 3. Experiment with Models and Datasets

You can define your own GNN models or datasets by following the modular structure provided in the repository. Check the `models/` and `datasets/` directories for more details.

## Repository Structure

```plaintext
MultiOmicsGraphEmbedding/
│
├── datasets/           # Contains dataset preprocessing and loading scripts
├── examples/           # Example scripts for experimentation
├── models/             # Implementation of various GNN models
├── utils/              # Utility functions for metrics, evaluation, etc.
├── requirements.txt    # Python dependencies
├── LICENSE             # License file
└── README.md           # Project documentation (this file)
```

## Contributing

We welcome contributions to the MultiOmicsGraphEmbedding project! If you have ideas for new features, bug fixes, or improvements, feel free to open an issue or submit a pull request.

### How to Contribute

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to your branch: `git push origin feature-name`.
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [DGL](https://www.dgl.ai/)
- [NetworkX](https://networkx.github.io/)

## Contact

For any questions or feedback, please feel free to create an issue in the repository or contact [JonnyTran](https://github.com/JonnyTran).
