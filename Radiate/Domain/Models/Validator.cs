using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Unsupervised;

namespace Radiate.Domain.Models;

public class Validator
{
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
        var predictions = new List<(float[] output, float[] target)>();
        
        foreach (var (inputs, answers) in data)
        {
            foreach (var (feature, target) in inputs.Zip(answers))
            {
                var (floats, _, _) = supervised.Predict(feature);
                var (_, loss) = _lossFunction(floats.ToTensor(), target);
            
                iterationLoss.Add(loss);
                predictions.Add((floats, target.Read1D()));   
            }
        }
        
        var classAcc = ClassificationAccuracy(predictions);
        var regAcc = RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Average(), classAcc, regAcc);
    }

    public Validation Validate(IUnsupervised unsupervised, List<Batch> data)
    {
        var iterationLoss = new List<float>();
        var predictions = new List<(float[] output, float[] target)>();
        
        foreach (var (inputs, answers) in data)
        {
            foreach (var (feature, target) in inputs.Zip(answers))
            {
                var (floats, _, _) = unsupervised.Predict(feature);
                var (_, loss) = _lossFunction(floats.ToTensor(), target);
            
                iterationLoss.Add(loss);
                predictions.Add((floats, target.Read1D()));   
            }
        }
        
        var classAcc = ClassificationAccuracy(predictions);
        var regAcc = RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Average(), classAcc, regAcc);
    }

    public static Epoch ValidateEpoch(List<float> errors, List<(float[] outputs, float[] targets)> predictions)
    {
        var classAccuracy = ClassificationAccuracy(predictions);
        var regressionAccuracy = RegressionAccuracy(predictions);

        return new Epoch(0, errors.Average(), classAccuracy, regressionAccuracy);
    }
    
    public static float ClassificationAccuracy(List<(float[] predictions, float[] targets)> outs)
    {
        var correctClasses = outs
            .Select(pair =>
            {
                var (first, second) = pair;
                var firstMax = first.ToList().IndexOf(first.Max());
                var secondMax = second.ToList().IndexOf(second.Max());

                return firstMax == secondMax ? 1f : 0f;
            })
            .Sum();

        return correctClasses / outs.Count;
    }
    
    public static float RegressionAccuracy(List<(float[] predictions, float[] targets)> outs)
    {
        var targetTotal = outs.Sum(tar => tar.targets.Sum());
        var absoluteDifference = outs
            .Select(pair => Math.Abs(pair.targets.First() - pair.predictions.First()))
            .Sum();

        return (targetTotal - absoluteDifference) / targetTotal;
    }
}