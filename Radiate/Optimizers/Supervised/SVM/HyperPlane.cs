using Radiate.Extensions;
using Radiate.Gradients;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Supervised.SVM.Info;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Supervised.SVM;

public class HyperPlane
{
    private const float Tolerance = 1e-7f;
    
    private readonly int _featureIndex;
    private readonly SVMInfo _svmInfo;
    private readonly GradientInfo _gradientInfo;
    private readonly Tensor _weights;
    private readonly Tensor _weightGradients;
    private readonly Stack<Tensor> _previousOutputs;
    
    public HyperPlane(SVMInfo svmInfo, int featureIndex, GradientInfo info)
    {
        var (height, width, depth) = svmInfo.FeatureShape;
        
        _featureIndex = featureIndex;
        _svmInfo = svmInfo;
        _gradientInfo = info;
        _previousOutputs = new Stack<Tensor>();
        _weights = Tensor.Random(height, width, depth) * .01f;
        _weightGradients = Tensor.Like(new Shape(height));
    }

    public HyperPlane(HyperPlaneWrap wrap)
    {
        _featureIndex = wrap.FeatureIndex;
        _gradientInfo = wrap.GradientInfo;
        _svmInfo = wrap.SVMInfo;
        _previousOutputs = new Stack<Tensor>();
        _weights = wrap.Weights;
        _weightGradients = Tensor.Like(wrap.Weights.Shape);
    }
    
    public Prediction Predict(Tensor input)
    {
        var approx = Tensor.Dot(input, _weights);
        var classification = approx >= 1 ? 1 : approx <= -1 ? -1 : 0;
        return new Prediction(new[] { approx }.ToTensor(), classification, approx);
    }

    public List<(Prediction prediction, Tensor target)> Feed(Tensor[] features, Tensor[] targets)
    {
        var result = new List<(Prediction prediction, Tensor target)>();
        foreach (var (x, y) in features.Zip(targets))
        {
            var prediction = Predict(x);
            
            _previousOutputs.Push(prediction.Result);
            result.Add((prediction, y));
        }

        return result;
    }
    
    public void Update(Tensor previousInput, Tensor previousTarget, Cost error, int epochCount)
    {
        var gradient = GradientFactory.Get(_gradientInfo);
        var previousOutput = _previousOutputs.Pop();
        var (height, _, _) = _weightGradients.Shape;
        
        var magnitude = Math.Abs(previousTarget.First() - _featureIndex) < Tolerance ? 1f : -1f;
        var plane = magnitude * previousOutput.Max();
        
        if (plane >= 1f)
        {
            for (var i = 0; i < height; i++)
            {
                _weightGradients[i] += 2f * _svmInfo.Lambda * _weights[i];
            }
        }
        else
        {
            for (var i = 0; i < height; i++)
            {
                _weightGradients[i] += (previousInput[i] * magnitude) - 2f * _svmInfo.Lambda * _weights[i] * error.Loss;
            }
        }
        
        _weights.Add(gradient.Calculate(_weightGradients, epochCount));
        _weightGradients.Zero();
    }

    public HyperPlaneWrap Save() => new()
    {
        FeatureIndex = _featureIndex,
        GradientInfo = _gradientInfo,
        SVMInfo = _svmInfo,
        Weights = _weights,
    };
}