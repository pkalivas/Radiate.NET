using Radiate.Domain.Loss;
using Radiate.Domain.Records;
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

    public async Task<T> Train(Func<Epoch, bool> trainFunc)
    {
        if (_optimizer is IPopulation population)
        {
            await population.Evolve(trainFunc);
            return _optimizer;
        }

        throw new Exception("Optimizer is not of type IPopulation and cannot be trained without Batched data.");
    }

    public async Task<T> Train(List<Batch> batches, Func<Epoch, bool> trainFunc)
    {
        if (_optimizer is ISupervised supervised)
        {
            await supervised.Train(batches, _lossFunction, trainFunc);
            return _optimizer;
        }
        
        throw new Exception("Optimizer is not of type ISupervised and cannot train through batches");
    }
}
