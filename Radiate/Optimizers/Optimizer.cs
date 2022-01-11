using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Callbacks.Resolver;
using Radiate.Domain.Extensions;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Models.Wraps;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.SVM;
using Radiate.Optimizers.TrainingSessions;
using Radiate.Optimizers.Unsupervised;
using Radiate.Optimizers.Unsupervised.Clustering;

namespace Radiate.Optimizers;

public class Optimizer<T> where T: class
{
    private readonly T _optimizer;
    private readonly TensorTrainSet _tensorTrainSet;
    private readonly Loss _loss;
    private readonly IEnumerable<ITrainingCallback> _callbacks;
    
    public Optimizer(T optimizer, TensorTrainSet tensorTrainSet, IEnumerable<ITrainingCallback> callbacks)
        : this(optimizer, tensorTrainSet, Loss.None, callbacks) { }

    public Optimizer(T optimizer, Loss loss = Loss.None, IEnumerable<ITrainingCallback> callbacks = null) 
        : this(optimizer, null, loss, callbacks) { }

    public Optimizer(T optimizer, TensorTrainSet tensorTrainSet, Loss loss = Loss.None,
        IEnumerable<ITrainingCallback> callbacks = null)
    {
        _optimizer = optimizer;
        _tensorTrainSet = tensorTrainSet;
        _loss = loss;
        _callbacks = callbacks ?? new List<ITrainingCallback>();
    }

    public async Task<T> Train() => await Train(_ => true);

    public async Task<T> Train(Func<Epoch, bool> trainFunc)
    {
        var lossFunction = _loss switch
        {
            Loss.None => LossFunctionResolver.Get(_optimizer),
            _ => LossFunctionResolver.Get(_loss)
        };
        
        var trainingSession = GetTrainingSession();
        var model = await trainingSession.Train<T>(_tensorTrainSet, lossFunction, trainFunc);
        
        foreach (var callback in CallbackResolver.Get<ITrainingCompletedCallback>(_callbacks))
        {
            await callback.CompleteTraining(this, trainingSession.Epochs, _tensorTrainSet);
        }

        return model;
    }

    public Prediction Predict(float[] input)
    {
        var processedInput = _tensorTrainSet.Process(input.ToTensor());
        return _optimizer switch
        {
            ISupervised supervised => supervised.Predict(processedInput),
            IUnsupervised unsupervised => unsupervised.Predict(processedInput),
            _ => throw new Exception("Cannot predict optimizer")
        };
    }

    public Validation ValidationScores()
    {
        var validator = new Validator(_loss switch
        {
            Loss.None => LossFunctionResolver.Get(_optimizer),
            _ => LossFunctionResolver.Get(_loss)
        });
        
        return _optimizer switch
        {
            ISupervised supervised => validator.Validate(supervised, _tensorTrainSet.TestingInputs),
            IUnsupervised unsupervised => validator.Validate(unsupervised, _tensorTrainSet.TestingInputs),
            _ => throw new Exception("Cannot validate model.")
        };
    }
    
    public OptimizerWrap Save() => new()
    {
        TensorOptions = _tensorTrainSet.TensorOptions,
        LossFunction = _loss,
        ModelWrap = _optimizer switch
        {
            MultiLayerPerceptron perceptron => perceptron.Save(),
            RandomForest forest => forest.Save(),
            SupportVectorMachine vectorMachine => vectorMachine.Save(),
            KMeans means => means.Save(),
            _ => throw new Exception("Cannot save optimizer")
        }
    };

    public static Optimizer<T> Load(OptimizerWrap wrap, IEnumerable<ITrainingCallback> callbacks = null) 
    {
        var modelWrap = wrap.ModelWrap;
        var trainSet = new TensorTrainSet(wrap.TensorOptions);
        
        switch (modelWrap.ModelType)
        {
            case ModelType.MultiLayerPerceptron:
            {
                var perceptron = new MultiLayerPerceptron(modelWrap);
                return new Optimizer<T>(perceptron as T, trainSet, wrap.LossFunction, callbacks);
            }
            case ModelType.RandomForest:
            {
                var perceptron = new RandomForest(modelWrap);
                return new Optimizer<T>(perceptron as T, trainSet, wrap.LossFunction, callbacks);
            }
            case ModelType.SVM:
            {
                var perceptron = new SupportVectorMachine(modelWrap);
                return new Optimizer<T>(perceptron as T, trainSet, wrap.LossFunction, callbacks);
            }
            case ModelType.KMeans:
            {
                var perceptron = new KMeans(modelWrap);
                return new Optimizer<T>(perceptron as T, trainSet, wrap.LossFunction, callbacks);
            }
            default:
                return null;
        }
    }

    private TrainingSession GetTrainingSession() => _optimizer switch
    {
        IPopulation population => new EvolutionTrainingSession(population, _callbacks),
        IUnsupervised unsupervised => new UnsupervisedTrainingSession(unsupervised, _callbacks),
        ISupervised supervised => new SupervisedTrainingSession(supervised, _callbacks),
        _ => throw new Exception("Cannot resolve training session.")
    };
}
