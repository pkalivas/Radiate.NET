using Radiate.Activations;
using Radiate.Gradients;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons;

public class MultiLayerPerceptron : ISupervised
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
    
    public MultiLayerPerceptron(ModelWrap wrap)
    {
        _layerInfo = new List<LayerInfo>();
        _layers = wrap.MultiLayerPerceptronWrap.LayerWraps
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

    public Shape Shape => _layers.Any() ? _layers.First().Shape : new Shape(0);
    
    public MultiLayerPerceptron AddLayer(LayerInfo layerInfo)
    {
        _layerInfo.Add(layerInfo);
        return this;
    }

    public List<(Prediction prediction, Tensor target)> Step(Tensor[] features, Tensor[] targets)
    {
        var predictions = new List<(Prediction, Tensor)>();
        
        foreach (var (x, y) in features.Zip(targets))
        {
            var prediction = PassForward(x);
            predictions.Add((prediction, y));
        }
        
        return predictions;
    }

    public void Update(List<Cost> errors, int epochCount)
    {
        foreach (var (passError, _) in errors.Select(pair => pair).Reverse())
        {
            var currentError = passError;
            for (var i = _layers.Count - 1; i >= 0; i--)
            {
                currentError = _layers[i].PassBackward(currentError);
            }
        }
                
        foreach (var layer in _layers)
        {
            layer.UpdateWeights(_gradientInfo, epochCount, errors.Count);
        }
    }

    public ModelWrap Save() => new()
    {
        ModelType = ModelType.MultiLayerPerceptron,
        MultiLayerPerceptronWrap = new()
        {
            LayerWraps = _layers.Select(layer => layer.Save()).ToList()
        }
    };

    public Prediction Predict(Tensor inputs)
    {
        var output = _layers.Aggregate(inputs, (current, layer) => layer.Predict(current));
        var maxIndex = output.ToList().IndexOf(output.Max());

        return new Prediction(output, maxIndex, output[maxIndex]);
    }
    
    public Prediction PassForward(Tensor input)
    {
        if (_layers.Any())
        {
            var result = _layers.Aggregate(input, (current, layer) => layer.FeedForward(current));
            return new Prediction(result, result.MaxIdx(), result.Max());
        }

        foreach (var layer in _layerInfo)
        {
            var shape = input.Shape;
            var newLayer = GetLayer(_layerInfo, layer, shape);
            input = newLayer.FeedForward(input);
            _layers.Add(newLayer);
        }

        return new Prediction(input, input.MaxIdx(), input.Max());
    }

    private static Layer GetLayer(List<LayerInfo> layers, LayerInfo info, Shape shape)
    {
        var (height, _, _) = shape;

        switch (info)
        {
            case DenseInfo denseInfo:
            {
                var activation = ActivationFunctionFactory.Get(denseInfo.Activation);
            
                return new Dense(new Shape(height, denseInfo.LayerSize, 0), activation);
            }
            case LSTMInfo lstm:
            {
                var cellActivation = ActivationFunctionFactory.Get(lstm.CellActivation);
                var hiddenActivation = ActivationFunctionFactory.Get(lstm.HiddenActivation);
            
                return new LSTM(new Shape(height, lstm.MemorySize, 0), cellActivation, hiddenActivation);
            }
            case MaxPoolInfo maxPool:
            {
                var previousLayer = layers[layers.IndexOf(info) - 1];
                var kernel = previousLayer switch
                {
                    ConvInfo cInfo => cInfo.Kernel,
                    _ => maxPool.Kernel
                };
            
                return new MaxPool(shape, kernel, maxPool.Stride);
            }
            case ConvInfo conv:
            {
                var activation = ActivationFunctionFactory.Get(conv.Activation);
                return new Conv(shape, conv.Kernel, conv.Stride, activation);
            }
            case DropoutInfo dropoutInfo: 
                return new Dropout(dropoutInfo.DropoutPercent);
            case FlattenInfo _: 
                return new Flatten(shape);
            default:
                throw new Exception($"Layer of {nameof(info)} does not exist");
        }
    }
    
}
