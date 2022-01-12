using Radiate.Losses;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Unsupervised;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers;

public class Validator
{
    private const float Tolerance = 0.0001f;
    
    private readonly LossFunction _lossFunction;

    public Validator(LossFunction lossFunction)
    {
        _lossFunction = lossFunction;
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

        var categoricalAccuracy = CategoricalAccuracy(predictions);
        var classAcc = ClassificationAccuracy(predictions);
        var regAcc = RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Sum(), classAcc, regAcc, categoricalAccuracy);
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

        var categoricalAccuracy = CategoricalAccuracy(predictions);
        var classAcc = ClassificationAccuracy(predictions);
        var regAcc = RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Sum(), classAcc, regAcc, categoricalAccuracy);
    }

    public static Epoch ValidateEpoch(List<float> errors, List<(Prediction outputs, Tensor targets)> predictions)
    {
        var categoricalAccuracy = CategoricalAccuracy(predictions);
        var classAccuracy = ClassificationAccuracy(predictions);
        var regressionAccuracy = RegressionAccuracy(predictions);

        return new Epoch(0, errors.Sum(), categoricalAccuracy, regressionAccuracy, classAccuracy);
    }
    
    private static float ClassificationAccuracy(List<(Prediction predictions, Tensor targets)> outs)
    {
        if (!outs.Any())
        {
            return 0f;
        }
        
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
        if (!outs.Any())
        {
            return 0f;
        }
        
        var targetTotal = outs.Sum(tar => tar.targets.Sum());
        var absoluteDifference = outs
            .Select(pair => Math.Abs(pair.targets.First() - pair.predictions.Confidence))
            .Sum();

        return (targetTotal - absoluteDifference) / targetTotal;
    }

    private static float CategoricalAccuracy(List<(Prediction prediction, Tensor targets)> predictions)
    {
        if (!predictions.Any())
        {
            return 0f;
        }
        
        var correctClasses = predictions
            .Sum(pair => Math.Abs(pair.targets.Max() - pair.prediction.Classification) < Tolerance ? 1f : 0f);

        return correctClasses / predictions.Count;
    }
}