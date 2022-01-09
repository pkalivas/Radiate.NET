using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.TrainingSessions;
using Radiate.Optimizers.Unsupervised;

namespace Radiate.Optimizers;

public class Optimizer<T>
{
    private readonly T _optimizer;
    private readonly LossFunction _lossFunction;
    private readonly TensorTrainSet _tensorTrainSet;
    private readonly IEnumerable<ITrainingCallback> _callbacks;
    
    public Optimizer(T optimizer, TensorTrainSet tensorTrainSet, IEnumerable<ITrainingCallback> callbacks)
        : this(optimizer, tensorTrainSet, Loss.Difference, callbacks) { }

    public Optimizer(T optimizer, Loss loss = new(), IEnumerable<ITrainingCallback> callbacks = null) 
        : this(optimizer, null, loss, callbacks) { }

    public Optimizer(T optimizer, TensorTrainSet tensorTrainSet, Loss loss = Loss.Difference, IEnumerable<ITrainingCallback> callbacks = null) 
        : this(optimizer, tensorTrainSet, LossFunctionResolver.Get(loss), callbacks) { }

    public Optimizer(T optimizer, TensorTrainSet tensorTrainSet, LossFunction lossFunction, IEnumerable<ITrainingCallback> callbacks)
    {
        _optimizer = optimizer;
        _tensorTrainSet = tensorTrainSet;
        _lossFunction = lossFunction;
        _callbacks = callbacks;
    }
    
    public async Task<T> Train() => await Train(_ => true);

    public async Task<T> Train(Func<Epoch, bool> trainFunc) =>
        await GetTrainingSession().Train<T>(_tensorTrainSet, _lossFunction, trainFunc);
    
    private TrainingSession GetTrainingSession() => _optimizer switch
    {
        IPopulation population => new EvolutionTrainingSession(population, _callbacks),
        IUnsupervised unsupervised => new UnsupervisedTrainingSession(unsupervised, _callbacks),
        ISupervised supervised => new SupervisedTrainingSession(supervised, _callbacks),
        _ => throw new Exception("Cannot resolve training session.")
    };
}
