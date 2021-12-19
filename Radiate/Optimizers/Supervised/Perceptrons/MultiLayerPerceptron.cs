using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;

namespace Radiate.Optimizers.Supervised.Perceptrons;

public class MultiLayerPerceptron : IOptimizer
{
    private readonly List<LayerInfo> _layerInfo;
    private readonly List<Layer> _layers;

    public MultiLayerPerceptron()
    {
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
    
    public MultiLayerPerceptron(Shape inputShape)
    {
        _layerInfo = new List<LayerInfo>();
        _layers = new List<Layer>();
    }

    public MultiLayerPerceptron AddLayer(LayerInfo layerInfo)
    {
        _layerInfo.Add(layerInfo);
        return this;
    }

    public Tensor Predict(Tensor inputs) =>
        _layers.Aggregate(inputs, (current, layer) => layer.Predict(current));

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
    
    public void PassBackward(Tensor errors, int epoch)
    {
        for (var i = _layers.Count - 1; i >= 0; i--)
        {
            errors = _layers[i].PassBackward(errors);
        }
    }

    public async Task Update(GradientInfo gradientInfo, int epoch) =>
        await Task.WhenAll(_layers
            .Select(async layer => await layer.UpdateWeights(gradientInfo, epoch)));

    public OptimizerWrap Save() => new()
    {
        OptimizerType = OptimizerType.MultiLayerPerceptron,
        MultiLayerPerceptronWrap = new()
        {
            LayerWraps = _layers.Select(layer => layer.Save()).ToList()
        }
    };

    private static Layer GetLayer(LayerInfo info, Shape shape)
    {
        var (height, width, depth) = shape;

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
