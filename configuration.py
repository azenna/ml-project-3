from types import SimpleNamespace

mlp_param_configurations = [
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
        "dropout": 0.2,
    },  # baseline
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
        "dropout": 0.5,
    },  # test if higher dropout better
    {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "optimizer": "adam",
        "dropout": 0.5,
    },  # test if lower learning rate better
    {
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "adam",
        "dropout": 0.5,
    },  # test if higher learning rate better
    {
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "adam",
        "dropout": 0.5,
    },  # test if batch_size = 64 better
    {
        "learning_rate": 0.001,
        "batch_size": 128,
        "optimizer": "adam",
        "dropout": 0.5,
    },  # test if batch_size = 128 better
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "sgd",
        "dropout": 0.2,
    },  # baseline sgd
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "sgd",
        "dropout": 0.5,
    },  # sgd test if higher dropout better
    {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "optimizer": "sgd",
        "dropout": 0.5,
    },  # sgd test if lower learning rate better
    {
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "sgd",
        "dropout": 0.5,
    },  # sgd test if higher learning rate better
    {
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "sgd",
        "dropout": 0.5,
    },  # sgd test if batch_size = 64 better
    {
        "learning_rate": 0.001,
        "batch_size": 128,
        "optimizer": "sgd",
        "dropout": 0.5,
    },  # sgd test if batch_size = 128 better
]

mlp_configurations = [
    SimpleNamespace(
        **{
            "arch": arch,
            "dataset": dataset,
            "tune": True,
            "num_epochs": 16,
            "weight_decay": 0.0,
            **config,
        }
    )
    for config in mlp_param_configurations
    for arch in ["shallow_mlp", "medium_mlp", "deep_mlp"]
    for dataset in ["mnist", "cifar"]
]

conv_param_configurations = [
    {"learning_rate": 0.001, "batch_size": 32, "weight_decay": 1e-4},  # baseline
    {
        "learning_rate": 0.01,
        "batch_size": 32,
        "weight_decay": 1e-4,
    },  # higher learning rate
    {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "weight_decay": 1e-4,
    },  # lower learning rate
    {
        "learning_rate": 0.001,
        "batch_size": 64,
        "weight_decay": 1e-4,
    },  # batch_size = 64
    {
        "learning_rate": 0.001,
        "batch_size": 128,
        "weight_decay": 1e-4,
    },  # batch_size = 128
    {
        "learning_rate": 0.001,
        "batch_size": 64,
        "weight_decay": 0.0,
    },  # weight_decay = 0
    {
        "learning_rate": 0.001,
        "batch_size": 64,
        "weight_decay": 5e-3,
    },  # weight_decay = 5e-3
]

conv_configurations = [
    SimpleNamespace(
        **{
            "arch": arch,
            "dataset": dataset,
            "tune": True,
            "num_epochs": 32,
            "dropout": 0.0,
            "optimizer": "adam",
            **config,
        }
    )
    for config in conv_param_configurations
    for arch in ["simple_conv", "enhanced_conv"]
    for dataset in ["mnist", "cifar"]
]

configurations = mlp_configurations + conv_configurations
