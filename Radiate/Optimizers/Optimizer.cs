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
    private readonly LossFunction _lossFunction;
    private readonly TensorTrainSet _tensorTrainSet;

    public Optimizer(T optimizer, Loss loss = new()) : this(optimizer, null, loss) { }

    public Optimizer(T optimizer, TensorTrainSet tensorTrainSet, Loss loss = Loss.Difference) 
        : this(optimizer, tensorTrainSet, LossFunctionResolver.Get(loss)) { }

    public Optimizer(T optimizer, TensorTrainSet tensorTrainSet, LossFunction lossFunction)
    {
        Model = optimizer;
        _tensorTrainSet = tensorTrainSet;
        _lossFunction = lossFunction;
    }

    public T Model { get; }

    public async Task Train(Func<Epoch, bool> trainFunc)
    {
        if (Model is IPopulation population)
        {
            await population.Evolve(trainFunc);
        }
        
        if (Model is IUnsupervised unsupervised)
        {
            var batch = _tensorTrainSet.TrainingInputs.Single();
            await unsupervised.Train(batch, _lossFunction, trainFunc);
        }
        
        if (Model is ISupervised supervised)
        {
            await supervised.Train(_tensorTrainSet.TrainingInputs, _lossFunction, trainFunc);
        }
    }

    public (Validation training, Validation testing) Validate()
    {
        var validator = new Validator(_lossFunction);

        switch (Model)
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
}
