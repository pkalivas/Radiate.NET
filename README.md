Common machine learning algorithm implementations. Extension of rust crate [radiate](https://github.com/pkalivas/radiate).

## Algorithms
1. **Random Forest**
2. **Support Vector Machine**
3. **Neural Networks**
    - Dense Layer
    - Dropout Layer
    - Flatten Layer
    - LSTM Layer
    - Convolutional Layer
    - MaxPooling Layer
4. **KMeans Clustering**
5. **Evolution Engine**
    - Implementation of [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

## Examples
- **Neural Network on Circles dataset:**
<img src="https://machinelearningmastery.com/wp-content/uploads/2017/12/Scatter-Plot-of-Circles-Test-Classification-Problem-1024x768.png" width="500px;" />

```c#
const int maxEpoch = 100;

var (inputs, targets) = await new Circles().GetDataSet();

var pair = new TensorTrainSet(inputs, targets)
    .TransformTargets(Norm.OHE)
    .TransformFeatures(Norm.Standardize)
    .Shuffle()
    .Split();

var mlp = new MultiLayerPerceptron(new GradientInfo { Gradient = Gradient.SGD })
    .AddLayer(new DenseInfo(32, Activation.ReLU))
    .AddLayer(new DenseInfo(pair.OutputCategories, Activation.Sigmoid));

var optimizer = new Optimizer<MultiLayerPerceptron>(mlp, pair, Loss.MSE, new List<ITrainingCallback>()
{
    new VerboseTrainingCallback(pair, maxEpoch, false),
    new ConfusionMatrixCallback()
});

await optimizer.Train(epoch => epoch.Index == maxEpoch);
```

- **Convolutional Neural Network on MNist dataset**
<img src="https://storage.googleapis.com/tfds-data/visualization/fig/mnist-3.0.1.png" width="200px">

```c#
const int featureLimit = 5000;
const int batchSize = 128;
const int maxEpochs = 10;

var (rawInputs, rawLabels) = await new Mnist(featureLimit).GetDataSet();

var pair = new TensorTrainSet(rawInputs, rawLabels)
    .Reshape(new Shape(28, 28, 1))
    .TransformFeatures(Norm.Image)
    .TransformTargets(Norm.OHE)
    .Batch(batchSize)
    .Split();

var neuralNetwork = new MultiLayerPerceptron()
    .AddLayer(new ConvInfo(64, 3))
    .AddLayer(new MaxPoolInfo(2))
    .AddLayer(new FlattenInfo())
    .AddLayer(new DenseInfo(64, Activation.Sigmoid))
    .AddLayer(new DenseInfo(pair.OutputCategories, Activation.SoftMax));

var optimizer = new Optimizer<MultiLayerPerceptron>(neuralNetwork, pair, new List<ITrainingCallback>
{
    new VerboseTrainingCallback(pair, maxEpochs),
    new ConfusionMatrixCallback()
});

await optimizer.Train(epoch => maxEpochs == epoch.Index);
```

## More
See the [examples](https://github.com/pkalivas/Radiate.NET/tree/main/Radiate.Examples/Examples) for how to use the API.

