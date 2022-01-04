using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Services;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised;

public class Optimizer : IOptimizer
{
    private readonly IOptimizer _optimizer;
    private readonly LossFunction _lossFunction;

    public Optimizer(IOptimizer optimizer, LossFunction lossFunction)
    {
        _optimizer = optimizer;
        _lossFunction = lossFunction;
    }
    
    public Optimizer(IOptimizer optimizer, Loss loss = Loss.Difference)
    {
        _optimizer = optimizer;
        _lossFunction = LossFunctionResolver.Get(loss);
    }

    public async Task Train(List<Batch> batches, Func<Epoch, bool> trainFunc) =>
        await Train(batches, _lossFunction, trainFunc);
    
    public async Task Train(List<Batch> batches, LossFunction lossFunction, Func<Epoch, bool> trainFunc) => 
        await _optimizer.Train(batches, lossFunction, trainFunc);

    public Prediction Predict(Tensor input) =>
        _optimizer.Predict(input);

    public OptimizerWrap Save() => _optimizer.Save();
    
    public Epoch Validate(List<Batch> batches)
    {
        var iterationLoss = new List<float>();
        var predictions = new List<(float[] output, float[] target)>();
        
        foreach (var (input, answer) in batches)
        {
            foreach (var (feature, target) in input.Zip(answer))
            {
                var (floats, _, _) = _optimizer.Predict(feature);
                var (_, loss) = _lossFunction(floats.ToTensor(), target);
            
                iterationLoss.Add(loss);
                predictions.Add((floats, target.Read1D()));   
            }
        }

        var classAcc = ValidationService.ClassificationAccuracy(predictions);
        var regAcc = ValidationService.RegressionAccuracy(predictions);

        return new Epoch(0, iterationLoss.Average(), classAcc, regAcc);
    }

}
