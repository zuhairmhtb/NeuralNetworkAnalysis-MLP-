# Multi-Layer Perceptron (MLP)

This file contains the implementation of simple MLP containing dense layers. Objective of this file is to compare the performance of models of these networks based on different features like:

1. Hyperparameters (Number of layers, Number of neurons per layer)

2. Activation functions

3. Optimization algorithm

4. Number of training dataset

5. Complexity of training dataset

6. Loss Functions

## Dataset:

The following datasets are used for comparison:

1. MNIST Handwritten digit dataset

2. CIFAR 100 dataset

3. (Text dataset)


```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
is_cuda_available = tf.test.is_gpu_available(cuda_only=True)
print(f'Cuda Available: {is_cuda_available}')
```

    WARNING:tensorflow:From <ipython-input-3-47c68bb476e6>:1: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.config.list_physical_devices('GPU')` instead.
    Cuda Available: True
    


```python
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

    Num GPUs Available:  1
    

## 1. Model


```python
def get_model(no_layers, neurons_per_layer, activation_function, output_function, optimisation_algorithm, loss_fn, total_inputs, total_classes, metrics=[]):
    """
    Returns a model with the specified hyper-parameters
    no_layers: Number of hidden layers in the network
    neurons_per_layer: Number of neurons per hidden layer
    activation_function: Activation function of the layers (except output layer)
    output_function: Acitvation function of the output layer
    optimization_algorithm: Optimization algorithm for the network
    loss_fn: Loss function for the network
    total_inputs: Number of neurons in the input layer
    total_classes: Number of neurons in the output layer
    """
    input_layer = tf.keras.layers.Input(shape=(total_inputs,))
    layer = None
    for i in range(no_layers):
        if i == 0:
            layer = tf.keras.layers.Dense(neurons_per_layer, activation=activation_function)(input_layer)
        else:
            layer = tf.keras.layers.Dense(neurons_per_layer, activation=activation_function)(layer)
    output_layer = tf.keras.layers.Dense(total_classes, activation=output_function)(layer)
    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(
        optimizer=optimisation_algorithm,
        loss=loss_fn,
        metrics=metrics
    )
    return model
```


```python
hidden_layers = [i for i in range(1, 15, 1)]
neurons_per_layer = [i for i in range(100, 550, 50)]
activation_functions = {
    'RELU': 'relu', 
    'Sigmoid': 'sigmoid', 
    'Softmax': 'softmax', 
    'Tanh': 'tanh'
}
optimization_algs = {
    'Adam': tf.keras.optimizers.Adam(0.001),
    'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.001),
    'Adadelta': tf.keras.optimizers.Adadelta(learning_rate=0.001),
    'RMSProp': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.001)
}

```

## 2. Dataset Preprocessing

### MNIST


```python
def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if np.max(x_train) > 1:
        x_train, x_test = x_train/(np.max(x_train) * 1.0), x_test/(np.max(x_test)*1.0)
    return x_train, y_train, x_test, y_test
```


```python
x_train, y_train, x_test, y_test = get_mnist_dataset()
print(f'Total Train dataset: {x_train.shape}, Train labels: {y_train.shape}, Test dataset: {x_test.shape}, Test labels: {y_test.shape}')
plt.figure(1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0], 'gray')
plt.show()
```

    Total Train dataset: (60000, 28, 28), Train labels: (60000,), Test dataset: (10000, 28, 28), Test labels: (10000,)
    


![png](output_12_1.png)


### CIFAR 100


```python
def get_cifar100_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    if np.max(x_train) > 1:
        x_train, x_test = x_train/(np.max(x_train) * 1.0), x_test/(np.max(x_test)*1.0)
    return x_train, y_train, x_test, y_test
```


```python
x_train, y_train, x_test, y_test = get_cifar100_dataset()
print(f'Total Train dataset: {x_train.shape}, Train labels: {y_train.shape}, Test dataset: {x_test.shape}, Test labels: {y_test.shape}')
plt.figure(1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0], 'gray')
plt.show()
```

    Total Train dataset: (50000, 32, 32, 3), Train labels: (50000, 1), Test dataset: (10000, 32, 32, 3), Test labels: (10000, 1)
    


![png](output_15_1.png)


## 3. Train Model


```python
def train_model(model, x_train, x_test, y_train, y_test, verbose=False, desc='', show_every=10, epochs=50, batch_size=32):
    accuracies = []
    train_accuracies = []
    losses = []
    for i in tqdm(range(epochs), desc=desc):
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_preds = np.argmax(model.predict(x_test), axis=-1)

        loss = np.mean(history.history['loss'])
        train_acc = np.mean(history.history['sparse_categorical_accuracy'])
        accuracy = accuracy_score(y_test, y_preds)

        accuracies.append(accuracy)
        losses.append(loss)
        train_accuracies.append(train_acc)
        if verbose and (i % show_every == 0):
            print(f'Epoch {i+1}, Loss: {loss}, Acc: {accuracy}, Train Accuracy: {train_acc}')
    return accuracies, losses, train_accuracies
```

## 4. Comparison of Different Input Layers

1. Neurons per layer: 100

2. Activation Fuction: Relu

3. Output activation: Softmax

4. Optimization: Adam

5. Loss: Sparse Categorical CrossEntropy

6. Batch Size: 32

7. Epochs: 50

#### MNIST


```python
x_train, y_train, x_test, y_test = get_mnist_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
for i in hidden_layers:
    time_to_train_x.append(f'Layers: {i}')

for i in hidden_layers:
    model = get_model(
        no_layers=i,
        neurons_per_layer=100,
        activation_function=activation_functions['RELU'],
        output_function='softmax',
        optimisation_algorithm=optimization_algs['Adam'],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Layers: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[i-1], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[i-1], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[i-1], label=str(i))

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Number of layers')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```

    Layers: 1: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [10:34<00:00, 12.68s/it]
    Layers: 2: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [11:05<00:00, 13.31s/it]
    Layers: 3: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [12:19<00:00, 14.78s/it]
    Layers: 4: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [12:44<00:00, 15.30s/it]
    Layers: 5: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [13:58<00:00, 16.77s/it]
    Layers: 6: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [14:20<00:00, 17.22s/it]
    Layers: 7: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [15:25<00:00, 18.51s/it]
    Layers: 8: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [15:53<00:00, 19.06s/it]
    Layers: 9: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [16:46<00:00, 20.13s/it]
    Layers: 10: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [17:26<00:00, 20.93s/it]
    Layers: 11: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [18:04<00:00, 21.68s/it]
    Layers: 12: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [18:54<00:00, 22.69s/it]
    Layers: 13: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [19:27<00:00, 23.35s/it]
    Layers: 14: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [20:33<00:00, 24.68s/it]
    


![png](output_21_1.png)


#### CIFAR-100


```python
x_train, y_train, x_test, y_test = get_cifar100_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
tf.debugging.set_log_device_placement(False)
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
for i in hidden_layers:
    time_to_train_x.append(f'Layers: {i}')

for i in hidden_layers:
    model = get_model(
        no_layers=i,
        neurons_per_layer=100,
        activation_function=activation_functions['RELU'],
        output_function='softmax',
        optimisation_algorithm=optimization_algs['Adam'],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Layers: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[i-1], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[i-1], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[i-1], label=str(i))

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Number of layers')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```

    Layers: 1: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [14:11<00:00, 17.03s/it]
    Layers: 2: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [14:45<00:00, 17.72s/it]
    Layers: 3: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [15:04<00:00, 18.09s/it]
    Layers: 4: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [15:59<00:00, 19.20s/it]
    Layers: 5: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [17:02<00:00, 20.46s/it]
    Layers: 6: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [17:03<00:00, 20.47s/it]
    Layers: 7: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [17:50<00:00, 21.40s/it]
    Layers: 8: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [17:57<00:00, 21.55s/it]
    Layers: 9: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [19:11<00:00, 23.03s/it]
    Layers: 10: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [20:10<00:00, 24.21s/it]
    Layers: 11: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [20:24<00:00, 24.50s/it]
    Layers: 12: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [20:17<00:00, 24.35s/it]
    Layers: 13: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [21:07<00:00, 25.36s/it]
    Layers: 14: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [23:02<00:00, 27.64s/it]
    


![png](output_24_1.png)


## 5. Comparison of Different Number of Neurons

1. Number of layers: 6 (Optimum accuracy layer for both dataset found above)

2. Activation Fuction: Relu

3. Output activation: Softmax

4. Optimization: Adam

5. Loss: Sparse Categorical CrossEntropy

6. Batch Size: 32

7. Epochs: 50

#### MNIST


```python
x_train, y_train, x_test, y_test = get_mnist_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
for i in neurons_per_layer[::-1]:
    time_to_train_x.append(f'Neurons: {i}')
    
count = 0
for i in neurons_per_layer[::-1]:
    model = get_model(
        no_layers=6,
        neurons_per_layer=i,
        activation_function=activation_functions['RELU'],
        output_function='softmax',
        optimisation_algorithm=optimization_algs['Adam'],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Neurons: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[count], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[count], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[count], label=str(i))
    
    count += 1

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Number of Neurons')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```

    Neurons: 500: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:52<00:00, 21.46s/it]
    Neurons: 450: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:29<00:00, 20.99s/it]
    Neurons: 400: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:27<00:00, 20.96s/it]
    Neurons: 350: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:12<00:00, 20.64s/it]
    Neurons: 300: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:04<00:00, 20.48s/it]
    Neurons: 250: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:08<00:00, 20.58s/it]
    Neurons: 200: 100%|████████████████████████████████████████████████████████████████████| 50/50 [16:01<00:00, 19.23s/it]
    Neurons: 150: 100%|████████████████████████████████████████████████████████████████████| 50/50 [14:39<00:00, 17.59s/it]
    Neurons: 100: 100%|████████████████████████████████████████████████████████████████████| 50/50 [14:41<00:00, 17.63s/it]
    


![png](output_28_1.png)


#### CIFAR 100


```python
x_train, y_train, x_test, y_test = get_cifar100_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
for i in neurons_per_layer[::-1]:
    time_to_train_x.append(f'Neurons: {i}')

count = 0
for i in neurons_per_layer[::-1]:
    model = get_model(
        no_layers=6,
        neurons_per_layer=i,
        activation_function=activation_functions['RELU'],
        output_function='softmax',
        optimisation_algorithm=optimization_algs['Adam'],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Neurons: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[count], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[count], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[count], label=str(i))
    
    count += 1

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Number of neurons')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```

    Neurons: 500: 100%|████████████████████████████████████████████████████████████████████| 50/50 [21:20<00:00, 25.62s/it]
    Neurons: 450: 100%|████████████████████████████████████████████████████████████████████| 50/50 [21:20<00:00, 25.62s/it]
    Neurons: 400: 100%|████████████████████████████████████████████████████████████████████| 50/50 [20:45<00:00, 24.91s/it]
    Neurons: 350: 100%|████████████████████████████████████████████████████████████████████| 50/50 [20:34<00:00, 24.70s/it]
    Neurons: 300: 100%|████████████████████████████████████████████████████████████████████| 50/50 [20:38<00:00, 24.78s/it]
    Neurons: 250: 100%|████████████████████████████████████████████████████████████████████| 50/50 [19:57<00:00, 23.95s/it]
    Neurons: 200: 100%|████████████████████████████████████████████████████████████████████| 50/50 [18:55<00:00, 22.72s/it]
    Neurons: 150: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:47<00:00, 21.35s/it]
    Neurons: 100: 100%|████████████████████████████████████████████████████████████████████| 50/50 [17:16<00:00, 20.73s/it]
    


![png](output_31_1.png)


## 6. Comparison of Different activations

1. Number of layers: 6 (Optimum accuracy layer for both dataset found above)

2. Number of Neurons: 250 (Optimum number of neurons for both dataset found above)

3. Output activation: Softmax

4. Optimization: Adam

5. Loss: Sparse Categorical Crossentropy

6. Batch Size: 32

7. Epochs: 50

#### MNIST


```python
x_train, y_train, x_test, y_test = get_mnist_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
activations = activation_functions.keys()
for i in activations:
    time_to_train_x.append(f'Activation: {i}')

count = 0
for i in activations:
    model = get_model(
        no_layers=6,
        neurons_per_layer=250,
        activation_function=activation_functions[i],
        output_function='softmax',
        optimisation_algorithm=optimization_algs['Adam'],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Activation: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[count], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[count], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[count], label=str(i))
    
    count += 1

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Activations')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```

    Activation: RELU: 100%|████████████████████████████████████████████████████████████████| 50/50 [17:11<00:00, 20.63s/it]
    Activation: Sigmoid: 100%|█████████████████████████████████████████████████████████████| 50/50 [17:06<00:00, 20.53s/it]
    Activation: Softmax: 100%|█████████████████████████████████████████████████████████████| 50/50 [18:52<00:00, 22.65s/it]
    Activation: Tanh: 100%|████████████████████████████████████████████████████████████████| 50/50 [17:13<00:00, 20.67s/it]
    


![png](output_35_1.png)


#### CIFAR 100


```python
x_train, y_train, x_test, y_test = get_cifar100_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
activations = activation_functions.keys()
for i in activations:
    time_to_train_x.append(f'Activation: {i}')

count=0 
for i in activations:
    model = get_model(
        no_layers=6,
        neurons_per_layer=250,
        activation_function=activation_functions[i],
        output_function='softmax',
        optimisation_algorithm=optimization_algs['Adam'],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Activation: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[count], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[count], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[count], label=str(i))
    
    count += 1

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Activations')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```

    Activation: RELU: 100%|████████████████████████████████████████████████████████████████| 50/50 [20:40<00:00, 24.82s/it]
    Activation: Sigmoid: 100%|█████████████████████████████████████████████████████████████| 50/50 [20:20<00:00, 24.40s/it]
    Activation: Softmax: 100%|█████████████████████████████████████████████████████████████| 50/50 [20:56<00:00, 25.13s/it]
    Activation: Tanh: 100%|████████████████████████████████████████████████████████████████| 50/50 [20:29<00:00, 24.58s/it]
    


![png](output_38_1.png)


## 7. Comparison of Optimization Algorithm

1. Number of layers: 6 (Optimum accuracy layer for both dataset found above)

2. Number of Neurons: 250 (Optimum number of neurons for both dataset found above)

3. Output activation: Softmax

4. Activation: ?? (Optimum activation for both dataset found above)

5. Loss: Sparse Categorical Crossentropy

6. Batch Size: 32

7. Epochs: 50

#### MNIST


```python
x_train, y_train, x_test, y_test = get_mnist_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
opt_algs = optimization_algs.keys()
for i in opt_algs:
    time_to_train_x.append(f'Optimization: {i}')

count = 0
for i in opt_algs:
    model = get_model(
        no_layers=6,
        neurons_per_layer=250,
        activation_function='relu', # Edit this
        output_function='softmax',
        optimisation_algorithm=optimization_algs[i],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Optimization: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[count], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[count], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[count], label=str(i))
    
    count += 1

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Optimizations')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```

#### CIFAR 100


```python
x_train, y_train, x_test, y_test = get_cifar100_dataset()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
```


```python
colors = [
    '#eb4034',
    '#eb7a34',
    '#eba534',
    '#ebdf34',
    '#c0eb34',
    '#34eb6b',
    '#34ebb1',
    '#34c3eb',
    '#9634eb',
    '#d934eb',
    '#eb34a2',
    '#eb3462',
    '#7a0909',
    '#5c5c06',
    '#06425c',
    '#000000'
]
plt.figure(1, figsize=(20, 30))
time_to_train = []
time_to_train_x = []
opt_algs = optimization_algs.keys()
for i in opt_algs:
    time_to_train_x.append(f'Optimization: {i}')

count = 0
for i in opt_algs:
    model = get_model(
        no_layers=6,
        neurons_per_layer=250,
        activation_function='relu', # Edit this
        output_function='softmax',
        optimisation_algorithm=optimization_algs[i],
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        total_inputs=x_train.shape[1],
        total_classes=np.unique(y_train).size,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    start = time.time()
    a, l, ta = train_model(model, x_train, x_test, y_train, y_test, desc=f'Optimization: {i}', show_every=2, batch_size=32, epochs=50)
    end = time.time()
    time_to_train.append(end-start)
    plt.subplot(4, 1, 1)
    plt.plot(a, color=colors[count], label=str(i))

    plt.subplot(4, 1, 2)
    plt.plot(l,color=colors[count], label=str(i))


    plt.subplot(4, 1, 3)
    plt.plot(ta,color=colors[count], label=str(i))
    
    count += 1

plt.subplot(4, 1, 1)
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Training Accuracy')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.title('Time to Train')
plt.bar(time_to_train_x, time_to_train, width=0.4)
plt.xlabel('Optimizations')
plt.ylabel('Seconds')
plt.grid()

plt.show()
```
