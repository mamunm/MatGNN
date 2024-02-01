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

## Usage

First, you need to define some global variables that will be used by both the data loader and the trainer:

```python
PREC = "64"
BATCH_SIZE = 64
DEVICE = "gpu"
ASE_DB_LOC = "path/to/database"
```

Here, `PREC` can be `"64", "32", or "16"` to specify the floating point precision. `Batch_SIZE` defines the batch size for both training and validation data loader. `DEVICE` defines the device type which can be eiher `"cpu" or "gpu"`. `ASE_DB_LOC` indicates the location of the ASE database.

After defining the global variables, we can now define the data parameters:

For `AtomGraph` module:

```python
atomgraph_params = AtomGraphParameters(
    feature_type="atomic_number",
    self_loop=True,
    graph_radius = 5
    max_neighbors = 5
    edge_feature = True
    edge_resolution = 50
    add_node_degree = True
    )
```

Here, `feature_type` can be `"atomic_number" or "atomic_symbol"` to specify the feature type. `self_loop` indicates whether to use self-loop or not. `graph_radius` can be used to limit the radius of the graph constructor to limit the number of neighbor. `max_neighbors` indicates the maximum number of neighbors to include that lie within the graph radius. `edge_feature` indicates whether to add edge feature or not. `edge_resolution` indicates the fineness of the edge feature constructor. `add_node_degree` indicates whether to add node degree to node feature.

After defining the feature constructor parameters, we can now incorporate that into the dataset parameters.

```python
ds_params = DatasetParameters(
    feature_type="AtomGraph",
    ase_db_loc=ASE_DB_LOC,
    target="hof",
    dtype=PREC,
    extra_parameters=atomgraph_params)
```

In the dataset parameters, the feature type can be `"AtomGraph", "SOAP", "CM", or "SM"` to specify the feature constructor type. `target` can be used to use a specific column of the database as the target.

Now, we can piece them altogether to define the data module parameters which will be used to create the training and validation dataloader.

```python
dm_params = DataModuleParameters(
    in_memory=True,
    dataset_params=ds_params,
    batch_size=BATCH_SIZE,
)
```

Data can either reside in memory or in file system (loaded on the fly). Now, we can initiate the data module to get some data parameters that will be needed to define the model parameters.

```python
dm = MaterialsGraphDataModule(dm_params)
dm.setup()
n_features = dm.dataset.num_features
n_edge_features = dm.dataset.num_edge_features
```

Now, we are ready to define the model parameters:

```python
model_params = GraphConvolutionParameters(
    n_features=n_features,
    n_edge_features=n_edge_features,
    batch_size=BATCH_SIZE,
    pre_hidden_size=130,
    post_hidden_size=120,
    gcn_hidden_size=150,
    n_gcn=4,
    n_pre_gcn_layers = 2,
    n_post_gcn_layers = 2,
    gcn_type="schnet",
    pool="max",
    dtype=PREC,
    device=DEVICE,
)
```

Here, `gcn_type` can be `"gcn", "cgcnn", or "schnet"`. To use message passing NN, you can similarly define the `MPNNParameters`.

We can then pass the model parameters to the matgnn constructor:

```python
mg_params = MatGNNParameters(
    model_params=model_params,
    optimizer="adam"
    )

mg = MatGNN(params=mg_params)
```

Finally, we can intialize the pytorch lightning trainer to train the model:

```python
trainer = pl.Trainer(accelerator=DEVICE,
                     max_epochs=200,
                     precision=PREC)

trainer.fit(
    model=mg,
    train_dataloaders=dm.train_dataloader(),
    val_dataloaders=dm.val_dataloader()
    )
```

Running this script will print the training statistics on the console.

## TODO

```[tasklist]
- [ ] MEGNET
- [ ] Test metrics
- [ ] logger
- [ ] saving and loading
- [ ] continue training from checkpoint
- [ ] Hyperparameter
- [ ] QM9
- [ ] CatHub surface data
```

## Author

- Osman Mamun, [Contact](mailto:mamun.che06@gmail.com), [Github](https://github.com/mamunm)
