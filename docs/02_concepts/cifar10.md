
## Datasets and DataLoaders

Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data. `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that subclass `torch.utils.data.Dataset` and implement functions specific to the particular data. They can be used to prototype and benchmark your model.


## Loading a Dataset

Here is an example of how to load the [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) dataset from TorchVision. Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

We load the [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) with the following parameters:

- `root` is the path where the train/test data is stored,
    
- `train` specifies training or test dataset,
    
- `download=True` downloads the data from the internet if it’s not available at `root`.
    
- `transform` and `target_transform` specify the feature and label transformations


```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
```

output:
```python
  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:12, 366kB/s]
  1%|          | 229k/26.4M [00:00<00:37, 698kB/s]
  3%|▎         | 918k/26.4M [00:00<00:11, 2.15MB/s]
 14%|█▍        | 3.67M/26.4M [00:00<00:03, 7.44MB/s]
 36%|███▋      | 9.63M/26.4M [00:00<00:00, 16.9MB/s]
 56%|█████▋    | 14.9M/26.4M [00:00<00:00, 24.9MB/s]
 70%|██████▉   | 18.4M/26.4M [00:01<00:00, 23.7MB/s]
 88%|████████▊ | 23.3M/26.4M [00:01<00:00, 29.5MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 19.7MB/s]

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 331kB/s]

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|▏         | 65.5k/4.42M [00:00<00:11, 369kB/s]
  4%|▍         | 197k/4.42M [00:00<00:07, 586kB/s]
 18%|█▊        | 786k/4.42M [00:00<00:01, 1.84MB/s]
 73%|███████▎  | 3.21M/4.42M [00:00<00:00, 6.49MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.20MB/s]

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 50.0MB/s]
```

## Iterating and Visualizing the Dataset

We can index `Datasets` manually like a list: `training_data[index]`. We use `matplotlib` to visualize some samples in our training data.

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

output:
```python
The output is labelled dataset images
```


## Creating a Custom Dataset for your files

A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. Take a look at this implementation; the FashionMNIST images are stored in a directory `img_dir`, and their labels are stored separately in a CSV file `annotations_file`.

In the next sections, we’ll break down what’s happening in each of these functions.


```python
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```


### `__init__`

The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the images, the annotations file, and both transforms (covered in more detail in the next section).

The labels.csv file looks like:

```python
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### `__len__`

The __len__ function returns the number of samples in our dataset.

Example:

```python
def __len__(self):
    return len(self.img_labels)
```

### `__getitem__`

The __getitem__ function loads and returns a sample from the dataset at the given index `idx`. Based on the index, it identifies the image’s location on disk, converts that to a tensor using `decode_image`, retrieves the corresponding label from the csv data in `self.img_labels`, calls the transform functions on them (if applicable), and returns the tensor image and corresponding label in a tuple.

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = decode_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```


## Preparing your data for training with DataLoaders

The `Dataset` retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s `multiprocessing` to speed up data retrieval.

`DataLoader` is an iterable that abstracts this complexity for us in an easy API.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## Iterate through the DataLoader

We have loaded that dataset into the `DataLoader` and can iterate through the dataset as needed. Each iteration below returns a batch of `train_features` and `train_labels` (containing `batch_size=64` features and labels respectively). Because we specified `shuffle=True`, after we iterate over all batches the data is shuffled (for finer-grained control over the data loading order, take a look at [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)).\

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

it will show the image along with the output:

```python
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 1
```

# Training a Classifier

This is it. You have seen how to define neural networks, compute loss and make updates to the weights of the network.

Now you might be thinking,

## What about data?

Generally, when you have to deal with image, text, audio or video data, you can use standard python packages that load data into a numpy array. Then you can convert this array into a `torch.*Tensor`.

- For images, packages such as Pillow, OpenCV are useful
    
- For audio, packages such as scipy and librosa
    
- For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful
    

Specifically for vision, we have created a package called `torchvision`, that has data loaders for common datasets such as ImageNet, CIFAR10, MNIST, etc. and data transformers for images, viz., `torchvision.datasets` and `torch.utils.data.DataLoader`.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset. It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

## Training an image classifier

We will do the following steps in order:

1. Load and normalize the CIFAR10 training and test datasets using `torchvision`
    
2. Define a Convolutional Neural Network
    
3. Define a loss function
    
4. Train the network on the training data
    
5. Test the network on the test data
    

### 1. Load and normalize CIFAR10

Using `torchvision`, it’s extremely easy to load CIFAR10.

```python
import torch
import torchvision
from torchvision.transforms import v2
```

The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].

```python
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

output:
```python
  0%|          | 0.00/170M [00:00<?, ?B/s]
  0%|          | 524k/170M [00:00<00:32, 5.23MB/s]
  3%|▎         | 4.62M/170M [00:00<00:06, 26.2MB/s]
  6%|▌         | 9.67M/170M [00:00<00:04, 37.2MB/s]
  8%|▊         | 14.3M/170M [00:00<00:03, 40.8MB/s]
 11%|█         | 18.4M/170M [00:00<00:03, 39.5MB/s]
 13%|█▎        | 22.3M/170M [00:00<00:03, 38.8MB/s]
 15%|█▌        | 26.2M/170M [00:00<00:03, 38.0MB/s]
 18%|█▊        | 30.1M/170M [00:00<00:03, 37.6MB/s]
 20%|█▉        | 33.8M/170M [00:00<00:03, 37.2MB/s]
 22%|██▏       | 37.6M/170M [00:01<00:03, 36.8MB/s]
 24%|██▍       | 41.3M/170M [00:01<00:03, 36.6MB/s]
 26%|██▋       | 45.0M/170M [00:01<00:03, 36.2MB/s]
 29%|██▊       | 48.6M/170M [00:01<00:03, 35.6MB/s]
 31%|███       | 52.2M/170M [00:01<00:03, 35.7MB/s]
 33%|███▎      | 55.8M/170M [00:01<00:03, 35.2MB/s]
 35%|███▍      | 59.3M/170M [00:01<00:03, 34.2MB/s]
 37%|███▋      | 62.9M/170M [00:01<00:03, 34.6MB/s]
 39%|███▉      | 66.4M/170M [00:01<00:03, 33.8MB/s]
 41%|████      | 70.1M/170M [00:01<00:02, 34.7MB/s]
 43%|████▎     | 73.8M/170M [00:02<00:02, 35.3MB/s]
 45%|████▌     | 77.5M/170M [00:02<00:02, 35.7MB/s]
 48%|████▊     | 81.1M/170M [00:02<00:02, 36.0MB/s]
 50%|████▉     | 84.7M/170M [00:02<00:02, 35.9MB/s]
 52%|█████▏    | 88.4M/170M [00:02<00:02, 35.9MB/s]
 54%|█████▍    | 92.0M/170M [00:02<00:02, 36.0MB/s]
 56%|█████▌    | 95.6M/170M [00:02<00:02, 35.4MB/s]
 58%|█████▊    | 99.2M/170M [00:02<00:02, 35.5MB/s]
 60%|██████    | 103M/170M [00:02<00:01, 35.7MB/s]
 63%|██████▎   | 107M/170M [00:02<00:01, 36.0MB/s]
 65%|██████▍   | 110M/170M [00:03<00:01, 36.1MB/s]
 67%|██████▋   | 114M/170M [00:03<00:01, 36.1MB/s]
 69%|██████▉   | 117M/170M [00:03<00:01, 36.2MB/s]
 72%|███████▏  | 123M/170M [00:03<00:01, 40.5MB/s]
 75%|███████▌  | 128M/170M [00:03<00:00, 44.6MB/s]
 78%|███████▊  | 133M/170M [00:03<00:00, 47.4MB/s]
 81%|████████▏ | 139M/170M [00:03<00:00, 49.3MB/s]
 85%|████████▍ | 144M/170M [00:03<00:00, 51.2MB/s]
 88%|████████▊ | 150M/170M [00:03<00:00, 52.0MB/s]
 91%|█████████ | 155M/170M [00:03<00:00, 52.9MB/s]
 94%|█████████▍| 161M/170M [00:04<00:00, 53.1MB/s]
 98%|█████████▊| 166M/170M [00:04<00:00, 53.7MB/s]
100%|██████████| 170M/170M [00:04<00:00, 39.9MB/s]
```

Let us show some of the training images, for fun.

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
```

the images and the output:

```python
bird plane cat bird
```

### Define a Convolutional Neural Network

Copy the neural network from the Neural Networks section before and modify it to take 3-channel images (instead of 1-channel images as it was defined).

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

### Define a Loss function and optimizer

Let’s use a Classification Cross-Entropy loss and SGD with momentum.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### Train the network

This is when things start to get interesting. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

output:
```python
[1,  2000] loss: 2.201
[1,  4000] loss: 1.852
[1,  6000] loss: 1.676
[1,  8000] loss: 1.571
[1, 10000] loss: 1.536
[1, 12000] loss: 1.476
[2,  2000] loss: 1.404
[2,  4000] loss: 1.366
[2,  6000] loss: 1.360
[2,  8000] loss: 1.341
[2, 10000] loss: 1.322
[2, 12000] loss: 1.287
Finished Training
```

to save our trained model:
```python
PATH = './cifar_net.pt'
torch.save(net.state_dict(), PATH)
```

### Test the network on the test data

We have trained the network for 2 passes over the training dataset. But we need to check if the network has learnt anything at all.

We will check this by predicting the class label that the neural network outputs, and checking it against the ground-truth. If the prediction is correct, we add the sample to the list of correct predictions.

Okay, first step. Let us display an image from the test set to get familiar.

```python
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
```

displaying images from the test set and output labels:
```python
GroundTruth: cat ship ship plane
```

Next, let’s load back in our saved model (note: saving and re-loading the model wasn’t necessary here, we only did it to illustrate how to do so):

```python
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
```

output:
```python
<All keys matched successfully>
```

The outputs are energies for the 10 classes. The higher the energy for a class, the more the network thinks that the image is of the particular class. So, let’s get the index of the highest energy:

```python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
```

output:
```python
Predicted: cat plane car ship
```

The results seem pretty good.

Let us look at how the network performs on the whole dataset.

```python
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

output:
```python
Accuracy of the network on the 10000 test images: 55 %
```

That looks way better than chance, which is 10% accuracy (randomly picking a class out of 10 classes). Seems like the network learnt something.

Hmmm, what are the classes that performed well, and the classes that did not perform well:

```python
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
```

output:
```python
Accuracy for class: plane is 62.1 %
Accuracy for class: car   is 69.8 %
Accuracy for class: bird  is 36.2 %
Accuracy for class: cat   is 30.8 %
Accuracy for class: deer  is 53.0 %
Accuracy for class: dog   is 49.6 %
Accuracy for class: frog  is 69.4 %
Accuracy for class: horse is 58.5 %
Accuracy for class: ship  is 69.6 %
Accuracy for class: truck is 59.2 %
```

Okay, so what next?

How do we run these neural networks on the GPU?

## Training on GPU

Just like how you transfer a Tensor onto the GPU, you transfer the neural net onto the GPU.

Let’s first define our device as the first visible cuda device if we have CUDA available:

```python
device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
```

output:
```python
cuda
```

The rest of this section assumes that `device` is a CUDA device.

Then these methods will recursively go over all modules and convert their parameters and buffers to CUDA tensors:

```python
net.to(device)
```

Remember that you will have to send the inputs and targets at every step to the GPU too:

```python
inputs, labels = data[0].to(device), data[1].to(device)
```

Why don’t I notice MASSIVE speedup compared to CPU? Because your network is really small.

**Exercise:** Try increasing the width of your network (argument 2 of the first `nn.Conv2d`, and argument 1 of the second `nn.Conv2d` – they need to be the same number), see what kind of speedup you get.

**Goals achieved**:

- Understanding PyTorch’s Tensor library and neural networks at a high level.
    
- Train a small neural network to classify images