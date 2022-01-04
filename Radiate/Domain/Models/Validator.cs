using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Services;
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
        
        var classAcc = ValidationService.ClassificationAccuracy(predictions);
        var regAcc = ValidationService.RegressionAccuracy(predictions);

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
                var (floats, _, _) = unsupervised.Predict(feature, _lossFunction);
                var (_, loss) = _lossFunction(floats.ToTensor(), target);
            
                iterationLoss.Add(loss);
                predictions.Add((floats, target.Read1D()));   
            }
        }
        
        var classAcc = ValidationService.ClassificationAccuracy(predictions);
        var regAcc = ValidationService.RegressionAccuracy(predictions);

        return new Validation(iterationLoss.Average(), classAcc, regAcc);
    }
}