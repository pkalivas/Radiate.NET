using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Services;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;

namespace Radiate.Optimizers.Supervised.Perceptrons;

public class MultiLayerPerceptron : IPredictor, ISupervised
{
    private readonly List<LayerInfo> _layerInfo;
    private readonly List<Layer> _layers;
    private readonly GradientInfo _gradientInfo;

    public MultiLayerPerceptron() : this(new GradientInfo()) { }

    public MultiLayerPerceptron(GradientInfo info)
    {
        _gradientInfo = info;
        _layerInfo = new List<LayerInfo>();
        _layers = new List<Layer>();
    }
    
    public MultiLayerPerceptron(MultiLayerPerceptronWrap wrap)
    {
        _layerInfo = new List<LayerInfo>();
        _layers = wrap.LayerWraps
            .Select(layerWrap => layerWrap.LayerType switch
            {
                LayerType.Conv => (Layer) new Conv(layerWrap.Conv),
                LayerType.Dense => new Dense(layerWrap.Dense),
                LayerType.Dropout => new Dropout(layerWrap.Dropout),
                LayerType.Flatten => new Flatten(layerWrap.Flatten),
                LayerType.LSTM => new LSTM(layerWrap.Lstm),
                LayerType.MaxPool => new MaxPool(layerWrap.MaxPool),
                _ => throw new Exception($"Layer {layerWrap.LayerType} is not loadable.")
            })
            .ToList();
    }
    
    public MultiLayerPerceptron AddLayer(LayerInfo layerInfo)
    {
        _layerInfo.Add(layerInfo);
        return this;
    }
    
    public async Task Train<T>(List<Batch<T>> batches, LossFunction lossFunction, Func<Epoch, bool> trainFunc)
    {
        var epochCount = 1;
        while (true)
        {
            var predictions = new List<(float[] output, float[] target)>();
            var epochErrors = new List<float>();

            foreach (var batch in batches)
            {
                var batchErrors = new List<Cost>();
                foreach (var (x, y) in batch.ReadPairs<Tensor>())
                {
                    var prediction = PassForward(x);
                    var cost = lossFunction(prediction, y);
                    
                    batchErrors.Add(cost);
                    predictions.Add((prediction.Read1D(), y.Read1D()));
                }

                foreach (var (passError, _) in batchErrors.Select(pair => pair).Reverse())
                {
                    PassBackward(passError);
                }

                await Update(epochCount);
                
                epochErrors.AddRange(batchErrors.Select(err => err.loss));
            }
            
            var classAccuracy = ValidationService.ClassificationAccuracy(predictions);
            var regAccuracy = ValidationService.RegressionAccuracy(predictions);
            var epoch = new Epoch(epochCount++, epochErrors.Average(), classAccuracy, regAccuracy);
            
            if (trainFunc(epoch))
            {
                break;
            }
        }
    }
    
    public OptimizerWrap Save() => new()
    {
        OptimizerType = OptimizerType.MultiLayerPerceptron,
        MultiLayerPerceptronWrap = new()
        {
            LayerWraps = _layers.Select(layer => layer.Save()).ToList()
        }
    };

    public Prediction Predict(Tensor inputs)
    {
        var output = _layers.Aggregate(inputs, (current, layer) => layer.Predict(current)).Read1D();
        var maxIndex = output.ToList().IndexOf(output.Max());

        return new Prediction(output, maxIndex, output[maxIndex]);
    }

    public Tensor PassForward(Tensor input)
    {
        if (_layers.Any())
        {
            return _layers.Aggregate(input, (current, layer) => layer.FeedForward(current));
        }

        foreach (var layer in _layerInfo)
        {
            var shape = input.Shape;
            var newLayer = GetLayer(layer, shape);
            input = newLayer.FeedForward(input);
            _layers.Add(newLayer);
        }

        return input;
    }
    
    private void PassBackward(Tensor errors)
    {
        for (var i = _layers.Count - 1; i >= 0; i--)
        {
            errors = _layers[i].PassBackward(errors);
        }
    }

    private async Task Update(int epoch) =>
        await Task.WhenAll(_layers.Select(layer => Task.Run(() => layer.UpdateWeights(_gradientInfo, epoch))));

    private static Layer GetLayer(LayerInfo info, Shape shape)
    {
        var (height, _, _) = shape;

        if (info is DenseInfo denseInfo)
        {
            var activation = ActivationFunctionFactory.Get(denseInfo.Activation);
            
            return new Dense(new Shape(height, denseInfo.LayerSize, 0), activation);
        }

        if (info is LSTMInfo lstm)
        {
            var cellActivation = ActivationFunctionFactory.Get(lstm.CellActivation);
            var hiddenActivation = ActivationFunctionFactory.Get(lstm.HiddenActivation);
            
            return new LSTM(new Shape(height, lstm.MemorySize, 0), cellActivation, hiddenActivation);
        }

        if (info is DropoutInfo dropoutInfo)
        {
            return new Dropout(dropoutInfo.DropoutPercent);
        }
        
        if (info is FlattenInfo _)
        {
            return new Flatten(shape);
        }

        if (info is MaxPoolInfo maxPool)
        {
            return new MaxPool(shape, maxPool.Kernel, maxPool.Stride);
        }

        if (info is ConvInfo conv)
        {
            var activation = ActivationFunctionFactory.Get(conv.Activation);
            return new Conv(shape, conv.Kernel, conv.Stride, activation);
        }
        
        throw new Exception($"Layer of {nameof(info)} does not exist");
    }
    
}
