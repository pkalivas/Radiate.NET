using Radiate.Extensions;
using Radiate.Gradients;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Supervised.SVM.Info;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Supervised.SVM;

public class SupportVectorMachine : ISupervised
{
    private readonly HyperPlane[] _hyperPlanes;
    private readonly Stack<Tensor> _previousInputs;
    private readonly Stack<Tensor> _previousTargets;

    public SupportVectorMachine(SVMInfo svmInfo, GradientInfo info)
    {
        var (_, numClasses, _) = svmInfo;
        
        _hyperPlanes = new HyperPlane[numClasses];
        _previousInputs = new Stack<Tensor>();
        _previousTargets = new Stack<Tensor>();
        
        for (var i = 0; i < _hyperPlanes.Length; i++)
        {
            _hyperPlanes[i] = new HyperPlane(svmInfo, i, info);
        }
    }

    public SupportVectorMachine(ModelWrap wrap)
    {
        var svm = wrap.SVMWrap;

        _previousInputs = new Stack<Tensor>();
        _previousTargets = new Stack<Tensor>();
        _hyperPlanes = svm.HyperPlanes.Select(plane => new HyperPlane(plane)).ToArray();
    }
    
    public Prediction Predict(Tensor input)
    {
        var planePredictions = new Prediction[_hyperPlanes.Length];
        Parallel.For(0, _hyperPlanes.Length, i =>
        {
            planePredictions[i] = _hyperPlanes[i].Predict(input);
        });
        
        var classes = planePredictions
            .Select(pred => pred.Confidence)
            .ToTensor();
        
        var classification = classes.MaxIdx();
        var confidence = classes[classification];

        return new Prediction(new[] { classes.Average() }.ToTensor(), classification, confidence);
    }

    public List<Step> Step(Tensor[] features, Tensor[] targets)
    {
        var result = new List<Step>();
        var separatorResults = new Prediction[_hyperPlanes.Length][];

        var startTime = DateTime.Now;
        Parallel.For(0, _hyperPlanes.Length, i =>
        {
            separatorResults[i] = _hyperPlanes[i].Feed(features, targets)
                .Select(pred => pred.prediction)
                .ToArray();
        });
        var stepTime = DateTime.Now - startTime;

        for (var i = 0; i < targets.Length; i++)
        {
            _previousInputs.Push(features[i]);
            _previousTargets.Push(targets[i]);
            
            var featurePreds = new List<Prediction>();
            for (var j = 0; j < _hyperPlanes.Length; j++)
            {
                featurePreds.Add(separatorResults[j][i]);
            }

            var predConf = featurePreds.Select(pred => pred.Confidence).ToList();
            var predMax = predConf.IndexOf(predConf.Max());
            var pred = featurePreds[predMax];
            var prediction = new Prediction(pred.Result, predMax, pred.Confidence);
            
            result.Add(new Step(prediction, targets[i], stepTime));
        }

        return result;
    }

    public void Update(List<Cost> errors, int epochCount)
    {
        foreach (var error in errors)
        {
            var previousInput = _previousInputs.Pop();
            var previousTarget = _previousTargets.Pop();
            foreach (var sep in _hyperPlanes)
            {
                sep.Update(previousInput, previousTarget, error, epochCount);
            }    
        }
    }

    public ModelWrap Save() => new()
    {
        ModelType = ModelType.SVM,
        SVMWrap = new()
        {
            HyperPlanes = _hyperPlanes.Select(sep => sep.Save()).ToList()
        }
    };
}