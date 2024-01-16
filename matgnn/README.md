# MatGNN: Graph Neural Network for Materials Science Research

A python library written in pytorch lightning for training GNN for materials science research.

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
- [ ] Replace data in data module with dataset parameters
```

## Author

- Osman Mamun, [Contact](mailto:mamun.che06@gmail.com), [Github](https://github.com/mamunm)
