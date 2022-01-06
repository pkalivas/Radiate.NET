using Radiate.Domain.Extensions;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Unsupervised;

namespace Radiate.Domain.Models;

public class Validator
{
    private const float Tolerance = 0.0001f;
    
    private readonly LossFunction _lossFunction;

    public Validator(LossFunction lossFunction)
    {
        _lossFunction = lossFunction;
    }

    public Validator(Loss.Loss loss = Loss.Loss.Difference)
    {
        _lossFunction = LossFunctionResolver.Get(loss);
    }
    
    public Validation Validate(ISupervised supervised, List<Batch> data)
    {
        var iterationLoss = new List<float>();
        var predictions = new List<(Prediction, Tensor)>();
        
        foreach (var (inputs, answers) in data)
        {
            foreach (var (feature, target) in inputs.Zip(answers))
            {
                var prediction = supervised.Predict(feature);
                var (_, loss) = _lossFunction(prediction.Result, target);
            
                iterationLoss.Add(loss);
                predictions.Add((prediction, target));   
            }
        }

        var acc = Accuracy(predictions);
        var classAcc = ClassificationAccuracy(predictions);
        var regAcc = RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Sum(), classAcc, regAcc, acc);
    }

    public Validation Validate(IUnsupervised unsupervised, List<Batch> data)
    {
        var iterationLoss = new List<float>();
        var predictions = new List<(Prediction output, Tensor target)>();
        
        foreach (var (inputs, answers) in data)
        {
            foreach (var (feature, target) in inputs.Zip(answers))
            {
                var prediction = unsupervised.Predict(feature);
                var (_, loss) = _lossFunction(prediction.Result, target);
            
                iterationLoss.Add(loss);
                predictions.Add((prediction, target));   
            }
        }

        var acc = Accuracy(predictions);
        var classAcc = ClassificationAccuracy(predictions);
        var regAcc = RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Sum(), classAcc, regAcc, acc);
    }

    public static Epoch ValidateEpoch(List<float> errors, List<(Prediction outputs, Tensor targets)> predictions)
    {
        var acc = Accuracy(predictions);
        var classAccuracy = ClassificationAccuracy(predictions);
        var regressionAccuracy = RegressionAccuracy(predictions);

        return new Epoch(0, errors.Sum(), classAccuracy, regressionAccuracy, acc);
    }
    
    private static float ClassificationAccuracy(List<(Prediction predictions, Tensor targets)> outs)
    {
        var correctClasses = outs
            .Select(pair =>
            {
                var (first, second) = pair;
                var secondMax = second.ToList().IndexOf(second.Max());
                
                return first.Classification == secondMax ? 1f : 0f;
            })
            .Sum();

        return correctClasses / outs.Count;
    }
    
    private static float RegressionAccuracy(List<(Prediction predictions, Tensor targets)> outs)
    {
        var targetTotal = outs.Sum(tar => tar.targets.Sum());
        var absoluteDifference = outs
            .Select(pair => Math.Abs(pair.targets.First() - pair.predictions.Confidence))
            .Sum();

        return (targetTotal - absoluteDifference) / targetTotal;
    }

    private static float Accuracy(List<(Prediction prediction, Tensor targets)> predictions)
    {
        var correctClasses = predictions
            .Sum(pair => Math.Abs(pair.targets.Max() - pair.prediction.Classification) < Tolerance ? 1f : 0f);

        return correctClasses / predictions.Count;
    }
}