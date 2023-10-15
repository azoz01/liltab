# liltab
liltab is a meta-learning package written in Python based on [1]. We implemented the inference network using PyTorch. In addition to the model, we provide a complete data loading and training API which in overall results in the possibility of end-to-end model creation. We also integrated our package with Tensorboard to monitor the training process in a real-time manner.

## Installation
To install package you need simply execute following statement:
``` bash
pip install liltab
```

## Why use liltab?
In case when you have few observations in your dataset and you want to have out of the box model which can be treated as a good starting point for further research, liltab is perfect solution. We provide model, which can be pretrained on any tabular data and then applied to specific tasks with no further training required. 

## How to use liltab?
Assume that you have plenty of `.csv` files with data with variable dimensionality. First of all you need to split it to three directories - `train`, `val` and `test`.  Next, you need to create data loaders from directory. The fastest approach is to use `ComposedDataLoaderFactory`. It hast plenty of parameters:
* `path` - path to data stored in csv format. Should contain only csv.
* `dataset_cls` - class which will encapsulate csv data to torch dataset. Either `PandasDataset` or `RandomFeaturesPandasDataset`
* `dataset_creation_args` -  Arguments passed to dataset constructors. Defaults to None.
* `loader_cls` - Class which will be used to load data from datasets. As for now only `FewShotDataLoader`.
* `dataloader_creation_args` - Arguments passed to dalaoader constructor. See `FewShotDataLoader` docstrings.
* `composed_dataloader_cls` - Class encapsulating all created dataloaders. Either `ComposedDataLoader` or `RepeatableOutputComposedDataLoader`.
* `batch_size` - size of batch which created dataloader will return.

Let's explain what particular classes do:
* `PandasDataset` - simple dataset encapsulating `.csv` data. Indexable class which on each indexing returns `X` and `y` tensors according to selected attributes and responses columns.
* `RandomFeaturesPandasDataset` - same as above, but on each indexing returns random subset of attributes and responses form data.
* `FewShotDataLoader` - iterable class which loads data with few-shot learning manner i.e. in each iterations returns `X_support`, `y_support`, `X_query`, `y_query`.
* `ComposedDataLoader` - iterable class coposing multiple instances of `FewShotDataLoader`. In each iteration returns observations from randomly selected lodaers.
* `RepeatableOutputComposedDataLoader` - Same as above, but in each iteration returns same sequence of examples. Useful during model validation.

Example of data loader creation:
``` Python
ComposedDataLoaderFactory.create_composed_dataloader_from_path(
    path=Path("train"),
    dataset_cls=RandomFeaturesPandasDataset,
    dataset_creation_args={},
    loader_cls=FewShotDataLoader,
    dataloader_creation_args={"support_size": 3, "query_size": 29},
    composed_dataloader_cls=ComposedDataLoader,
    batch_size=32,
)
```
Having datasets created, we can create model using `HeterogenousAttributesNetwork` using following parameters:
* `hidden_representation_size` - Size of hidden representation sizes i. e. all intermediate network outputs.
* `n_hidden_layers` - number hidden layers of networks used during inference.
* `hidden_size` - number of neurons per hidden layer in networks using during inference.
* `dropout_rate` - dropout rate of networks used during inference.
* `inner_activation_function` - inner activation function of networks used during inference.
* `output_activation_function` - output activation function of final network used during inference.
* `is_classifier` - if `True` then the output of the network will generate probabilities of classes for the query set.

Example of model creation:
``` Python
HeterogenousAttributesNetwork(
    hidden_representation_size=16,
    n_hidden_layers=1,
    hidden_size=16,
    dropout_rate=0.2,
    inner_activation_function=nn.ReLU(),
    output_activation_function=nn.Identity(),
    is_classifier=False
)
```

Finally we can create object responsible for training - `HeterogenousAttributesNetworkTrainer` using params:
* `n_epochs` - number of epochs to train
* `gradient_clipping` - if `True`, then gradient clipping is applied
* `learning_rate` - learning rate used during training,
* `weight_decay` - weight decay used during training,
* `early_stopping` - if `True` then early stopping is applied,
* `file_logger` - if `True` then logging to `.csv` file is used,
* `tb_logger` - if `True` then logging to Tensorboard is used,
* `model_checkpoints` - if `True` then model checkpoints are used,
* `results_path` - path to results directory.

Example of trainer creation:
``` Python
HeterogenousAttributesNetworkTrainer(
    n_epochs=100_000,
    gradient_clipping=False,
    learning_rate=1e-3,
    weight_decay=1e-4,
    early_stopping=True,
    file_logger=True,
    tb_logger=True,
    model_checkpoints=True,
    results_path=Path("sample_results"),
)
```

Finally to train model you need to call `train_and_test` method e. g.
``` Python
trainer.train_and_test(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)
```

**For complete examples of usage see `experiments` directory.**
## Dev
You need to have Python 3.10 and pip.

To get dependencies run: 
``` bash
pip install -r requirements.txt
```

To format and check code with linter run:
``` bash
make prepare_code
```

Run tests using following: 
``` bash
make run_tests
```

## Authors
Package was created as a result of thesis by
* **Antoni Zajko**,
* **Dawid Płudowski**.

Project co-ordinator and supervisor: **Anna Kozak**
## References
[1] Iwata, T. and Kumagai, A. (2020). Meta-learning from Tasks with
Heterogeneous Attribute Spaces. In Advances in Neural Information Processing Systems,
volume 33, pages 6053–6063. Curran Associates, Inc.
