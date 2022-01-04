using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Services;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Supervised;

namespace Radiate.Optimizers;

public class Optimizer<T>
{
    private readonly T  _optimizer;
    private readonly LossFunction _lossFunction;

    public Optimizer(T optimizer, LossFunction lossFunction)
    {
        _optimizer = optimizer;
        _lossFunction = lossFunction;
    }
    
    public Optimizer(T optimizer, Loss loss = Loss.Difference)
    {
        _optimizer = optimizer;
        _lossFunction = LossFunctionResolver.Get(loss);
    }

    public async Task<T> Evolve(Func<double, int, bool> trainFunc)
    {
        if (_optimizer is not IPopulation population)
        {
            throw new Exception("Optimizer is not of type IPopulation and cannot be trained through evolution.");
        }

        await population.Evolve(trainFunc);
        return _optimizer;
    }

    public async Task<T> Train<TB>(List<Batch<TB>> batches, Func<Epoch, bool> trainFunc)
    {
        if (_optimizer is not ISupervised supervised)
        {
            throw new Exception("Optimizer is not of type ISupervised and cannot train through batches");
        }
        
        await supervised.Train(batches, _lossFunction, trainFunc);
        return _optimizer;

    }
    
    public Validation Validate(List<Batch<Tensor>> data)
    {
        if (_optimizer is not IPredictor optimizer)
        {
            throw new Exception("Cannot validate optimizer");
        }

        var predictor = (IPredictor)optimizer;
        var iterationLoss = new List<float>();
        var predictions = new List<(float[] output, float[] target)>();
        
        foreach (var batch in data)
        {
            foreach (var (feature, target) in batch.ReadPairs<Tensor>())
            {
                var (floats, _, _) = predictor.Predict(feature);
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
