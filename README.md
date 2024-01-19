# MatGNN: Graph Neural Network Library for Materials Science Research

> A python library written in pytorch lightning to train GNN for materials science research.

Besides functionality to perform a typical GNN training, MatGNN's highlights are:

- Data module
- hyperparameter

## Installation

To create a virtual environment for MatGNN using conda or mamba, run the following command:

```bash
mamba env create -n matgnn -f environment.yml
```

Then you need to add the following to your run script:

```python
import sys
sys.path.append("/path/to/matgnn")
```

## TODO

```[tasklist]
Features
- [ ] SOAP
- [ ] ACSF
- [ ] Atomic graph
Models
- [ ] CGCNN
- [ ] GCN
- [ ] MEGNET
- [ ] MPNN
- [ ] SCHNET
Data
- [ ] Dataset

- [ ] Trainer
- [ ] Hyperparameter
```

## Author

- Osman Mamun, [Contact](mailto:mamun.che06@gmail.com), [Github](https://github.com/mamunm)
