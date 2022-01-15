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

## Features
1. Training callbacks similar to [Keras Callbacks](https://keras.io/api/callbacks/)
> Hook into the training loop with custom code. Create an object that implements any or all of the following:
> 1. IEpochStartedCallback
> 2. IBatchCompletedCallback
> 3. IEpochCompletedCallback
> 4. ITrainingCompletedCallback

2. Functional feature engineering with ```TensorTrainSet```
>1. Batch
>    - Set a batch size to train on.
>2. Layer
>    - Layer data by n rows.
>3. Split
>    - Split the data into a training set and testing set. Default is 75% split training, 25% testing.
>4. Reshape
>    - Reshape the row vector to a shape of (height, width, depth), useful for images.
>5. Pad
>    - Pad an image Tensor with n zeros.
>6. Shuffle
>    - Shuffle the rows of the dataset randomly.
>7. Kernel
>    - Add kernel transform for the features, possible options are *RBF*, *Polynomial*, and *Linear* (None).
>8. TransformFeatures
>    - Transform the feature data. Options are *Normalize*, *Standardize*, *OHE* (One Hot Encode), and *Image* (divide data point by 255).
>8. TransformTargets
>    - Transform the target data. Options same as above.
3. Model saving/loading to/from Json

### Feature Engineering
---
1. Batch
    - Set a batch size to train on.
2. Layer
    - Layer data by n rows.
3. Split
    - Split the data into a training set and testing set. Default is 75% split training, 25% testing.
4. Reshape
    - Reshape the row vector to a shape of (height, width, depth), useful for images.
5. Pad
    - Pad an image Tensor with n zeros.
6. Shuffle
    - Shuffle the rows of the dataset randomly.
7. Kernel
    - Add kernel transform for the features, possible options are *RBF*, *Polynomial*, and *Linear* (None).
8. TransformFeatures
    - Transform the feature data. Options are *Normalize*, *Standardize*, *OHE* (One Hot Encode), and *Image* (divide data point by 255).
8. TransformTargets
    - Transform the target data. Options same as above.

## Examples

**Convolutional Neural Network on MNist handwritten digets dataset**

<img src="https://camo.githubusercontent.com/01c057a753e92a9bc70b8c45d62b295431851c09cffadf53106fc0aea7e2843f/687474703a2f2f692e7974696d672e636f6d2f76692f3051493378675875422d512f687164656661756c742e6a7067" width="300px">

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

**Random Forest on Iris Flowers dataset**

<img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_dataset_scatterplot.svg" width="300px">

```c#
const int numTrees = 10;
const int maxDepth = 10;
const int minSampleSplit = 2;

var (rawFeatures, rawLabels) = await new IrisFlowers().GetDataSet();
var pair = new TensorTrainSet(rawFeatures, rawLabels)
    .Shuffle()
    .Split()
    .Batch(rawFeatures.Count);

var forest = new RandomForest(numTrees, new ForestInfo(minSampleSplit, maxDepth));
var optimizer = new Optimizer<RandomForest>(forest, pair, new List<ITrainingCallback>
{
    new VerboseTrainingCallback(pair),
    new ModelWriterCallback(),
    new ConfusionMatrixCallback()
});

await optimizer.Train();
```

## More
See the [examples](https://github.com/pkalivas/Radiate.NET/tree/main/Radiate.Examples/Examples) for how to use the API.

