# Simple Learning Pipeline using Jax

This repository contains code for a basic learning pipeline using Jax, a numerical computing library for machine learning research.

## Usage

To use the code in this repository, follow these steps:

1. Set hyperparameters:
```python
hp = Hyperparam()
hp.dims = [2, 10, 10, 1]
hp.lr = 0.001
hp.batch_size = 128
 
```

2. Load data:
```python
df = pd.read_csv("training_data/circle.csv")
dataset = NumpyDataset(df[["x", "y"]].to_numpy(), df["d"].to_numpy())
train_dataset, val_dataset = train_test_split(dataset, train_size=0.9, shuffle=True)

train_loader = data.DataLoader(
    train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=numpy_collate)
val_loader = data.DataLoader(
    val_dataset, batch_size=hp.batch_size, collate_fn=numpy_collate)
```

3. Create model and initialize parameters:
```python
model = get_mlp(hp)
key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (2,))
params = model.init(key2, x)
```

4. Train model and save checkpoints:
```python
tx = optax.adam(learning_rate=hp.lr)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
trained_state = trainer(
    state, train_loader, val_loader, l2_loss_fn,
    num_epochs=100, exp_str=hp.as_str())
    
save("model", trained_state, hp, force=True)
```

5. Load last checkpoint and use model:
```python
sdf_fn = get_mlp_by_path("./model")
sdf_fn(jnp.zeros(2))
```

## License
This project is licensed under the terms of the MIT license. 
