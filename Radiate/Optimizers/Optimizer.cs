using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Callbacks.Resolver;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Supervised;
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

    public T Model => _optimizer;

    public async Task<T> Train() => await Train(_ => true);

    public async Task<T> Train(Func<Epoch, bool> trainFunc)
    {
        if (_optimizer is IPopulation population)
        {
            await population.Evolve(trainFunc);
        }
        
        if (_optimizer is IUnsupervised unsupervised)
        {
            var data = _tensorTrainSet.TrainingFeatureInputs
                .SelectMany(batch => batch.Features.Select(row => row))
                .ToArray();
            unsupervised.Train(data, trainFunc);
        }
        
        if (_optimizer is ISupervised supervised)
        {
            TrainSupervised(supervised, trainFunc);
            // supervised.Train(_tensorTrainSet.TrainingInputs, _lossFunction, trainFunc);
        }

        return _optimizer;
    }

    public (Validation training, Validation testing) Validate()
    {
        var validator = new Validator(_lossFunction);

        switch (_optimizer)
        {
            case IUnsupervised unsupervised:
            {
                var trainValid = validator.Validate(unsupervised, _tensorTrainSet.TrainingInputs);
                var testValid = validator.Validate(unsupervised, _tensorTrainSet.TestingInputs);

                return (trainValid, testValid);
            }
            case ISupervised supervised:
            {
                var trainValid = validator.Validate(supervised, _tensorTrainSet.TrainingInputs);
                var testValid = validator.Validate(supervised, _tensorTrainSet.TestingInputs);

                return (trainValid, testValid);
            }
            default:
                throw new Exception($"Cannot validate optimizer type.");
        }
    }

    private void TrainSupervised(ISupervised supervisedModel, Func<Epoch, bool> trainFunc)
    {
        var trainSession = new SupervisedTrainingSession(supervisedModel, _callbacks);
        var batches = _tensorTrainSet.TrainingInputs;
        
        while (true)
        {
            var epoch = trainSession.Fit(batches, _lossFunction);

            if (trainFunc(epoch))
            {
                break;
            }
        }

        trainSession.CompleteTraining<T>();
    }
}
