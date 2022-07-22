using Radiate.Losses;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Interfaces;
using Radiate.Optimizers.Unsupervised;
using Radiate.Optimizers.Unsupervised.Interfaces;
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

    public Validation Validate(IOptimizerModel model, List<Batch> data)
    {
        var iterationLoss = new List<float>();
        var predictions = new List<Step>();
        
        foreach (var (inputs, answers) in data)
        {
            foreach (var (feature, target) in inputs.Zip(answers))
            {
                var startTime = DateTime.Now;
                var prediction = model switch
                {
                    IEvolved evolved => evolved.Predict(feature),
                    ISupervised supervised => supervised.Predict(feature),
                    IUnsupervised unsupervised => unsupervised.Predict(feature),
                    _ => throw new Exception("Cannot validate model")
                };
                
                var stepTime = DateTime.Now - startTime;
                var (_, loss) = _lossFunction(prediction.Result, target);
            
                iterationLoss.Add(loss);
                predictions.Add(new Step(prediction, target, stepTime));   
            }
        }

        var categoricalAccuracy = CategoricalAccuracy(predictions);
        var classAcc = ClassificationAccuracy(predictions);
        var regAcc = RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Sum(), classAcc, regAcc, categoricalAccuracy);
    }
    
    public static Epoch ValidateEpoch(List<float> errors, List<Step> predictions)
    {
        var categoricalAccuracy = CategoricalAccuracy(predictions);
        var classAccuracy = ClassificationAccuracy(predictions);
        var regressionAccuracy = RegressionAccuracy(predictions);
        var totalStepMillis = (double) predictions.Sum(step => step.Time.TotalMilliseconds) / predictions.Count;
        
        return new Epoch(0, errors.Sum(), categoricalAccuracy, 
            regressionAccuracy, classAccuracy, 0, default, default);
    }
    
    private static float ClassificationAccuracy(List<Step> outs)
    {
        if (!outs.Any())
        {
            return 0f;
        }
        
        var correctClasses = outs
            .Select(pair =>
            {
                var (first, second, _) = pair;
                var secondMax = second.ToList().IndexOf(second.Max());
                
                return first.Classification == secondMax ? 1f : 0f;
            })
            .Sum();

        return correctClasses / outs.Count;
    }
    
    private static float RegressionAccuracy(List<Step> outs)
    {
        if (!outs.Any())
        {
            return 0f;
        }
        
        var targetTotal = outs.Sum(tar => tar.Target.Sum());
        var absoluteDifference = outs
            .Select(pair => Math.Abs(pair.Target.First() - pair.Prediction.Confidence))
            .Sum();

        return (targetTotal - absoluteDifference) / targetTotal;
    }

    private static float CategoricalAccuracy(List<Step> predictions)
    {
        if (!predictions.Any())
        {
            return 0f;
        }
        
        var correctClasses = predictions
            .Sum(pair => Math.Abs(pair.Target.Max() - pair.Prediction.Classification) < Tolerance ? 1f : 0f);

        return correctClasses / predictions.Count;
    }
}