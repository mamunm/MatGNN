# MatGNN: Graph Neural Network Library for Materials Science

![MatGNN Logo](images/logo.png)

> A python library written in pytorch lightning to train GNN for materials science research.

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
- [ ] MEGNET
- [ ] Test metrics
- [ ] logger
- [ ] saving and loading
- [ ] continue training from checkpoint
- [ ] Hyperparameter
```

## Author

- Osman Mamun, [Contact](mailto:mamun.che06@gmail.com), [Github](https://github.com/mamunm)
