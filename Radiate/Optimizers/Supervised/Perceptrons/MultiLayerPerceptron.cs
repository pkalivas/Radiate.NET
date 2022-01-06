using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;

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
    
    public MultiLayerPerceptron(SupervisedWrap wrap)
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
    
    public void Train(List<Batch> batches, LossFunction lossFunction, Func<Epoch, bool> trainFunc)
    {
        var epochCount = 1;
        while (true)
        {
            var predictions = new List<(Prediction, Tensor)>();
            var epochErrors = new List<float>();

            foreach (var (inputs, answers) in batches)
            {
                var batchErrors = new List<Cost>();
                foreach (var (x, y) in inputs.Zip(answers))
                {
                    var prediction = PassForward(x, y);
                    var cost = lossFunction(prediction.Result, y);
                    
                    batchErrors.Add(cost);
                    predictions.Add((prediction, y));
                }

                foreach (var (passError, _) in batchErrors.Select(pair => pair).Reverse())
                {
                    PassBackward(passError);
                }
                
                foreach (var layer in _layers)
                {
                    layer.UpdateWeights(_gradientInfo, epochCount, predictions.Count);
                }
                
                epochErrors.AddRange(batchErrors.Select(err => err.Loss));
            }

            var epoch = Validator.ValidateEpoch(epochErrors, predictions);
            
            if (trainFunc(epoch with { Index = epochCount++ }))
            {
                break;
            }
        }
    }
    
    public SupervisedWrap Save() => new()
    {
        SupervisedType = SupervisedType.MultiLayerPerceptron,
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
    
    public Prediction PassForward(Tensor input, Tensor target)
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
    
    private void PassBackward(Tensor errors)
    {
        for (var i = _layers.Count - 1; i >= 0; i--)
        {
            errors = _layers[i].PassBackward(errors);
        }
    }
    
    private static Layer GetLayer(List<LayerInfo> layers, LayerInfo info, Shape shape)
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
            var previousLayer = layers[layers.IndexOf(info) - 1];
            var kernel = previousLayer switch
            {
                ConvInfo cInfo => cInfo.Kernel,
                _ => maxPool.Kernel
            };
            
            return new MaxPool(shape, kernel, maxPool.Stride);
        }

        if (info is ConvInfo conv)
        {
            var activation = ActivationFunctionFactory.Get(conv.Activation);
            return new Conv(shape, conv.Kernel, conv.Stride, activation);
        }
        
        throw new Exception($"Layer of {nameof(info)} does not exist");
    }
    
}
