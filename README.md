Common machine learning algorithms implemented from scratch. Extension of rust crate [radiate](https://github.com/pkalivas/radiate).

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
<img src="https://machinelearningmastery.com/wp-content/uploads/2017/12/Scatter-Plot-of-Circles-Test-Classification-Problem-1024x768.png" width="500px;">
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

## More
See the [examples](https://github.com/pkalivas/Radiate.NET/tree/main/Radiate.Examples/Examples) for how to use the API.

