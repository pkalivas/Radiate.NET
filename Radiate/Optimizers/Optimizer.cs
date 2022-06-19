﻿using Radiate.Callbacks.Interfaces;
using Radiate.Callbacks.Resolver;
using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Losses;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Neat;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.SVM;
using Radiate.Optimizers.TrainingSessions;
using Radiate.Optimizers.Unsupervised;
using Radiate.Optimizers.Unsupervised.Clustering;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers;

public class Optimizer<T> where T: class
{
    private readonly T _optimizer;
    private readonly TensorTrainSet _tensorTrainSet;
    private readonly Population<T> _population;
    private readonly Loss _loss;
    private readonly IEnumerable<ITrainingCallback> _callbacks;

    public Optimizer(Population<T> population, TensorTrainSet tensorTrainSet = null)
    {
        _population = population;
        _tensorTrainSet = tensorTrainSet?.Compile();
        _callbacks = new List<ITrainingCallback>();
    }

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
        Model = _optimizer;
    }
    
    private T Model { get; set; }

    public async Task<T> Train() => await Train(_ => Task.FromResult(true));

    public async Task<T> Train(Func<Epoch, bool> trainFunc) =>
        await Train(epoch => Task.Run(() => trainFunc(epoch)));

    public async Task<T> Train(Func<Epoch, Task<bool>> trainFunc)
    {
        var lossFunction = _loss switch
        {
            Loss.None => LossFunctionResolver.Get(_optimizer),
            _ => LossFunctionResolver.Get(_loss)
        };
        
        var trainingSession = GetTrainingSession();
        Model = await trainingSession.Train<T>(_tensorTrainSet, lossFunction, trainFunc);
        
        foreach (var callback in CallbackResolver.Get<ITrainingCompletedCallback>(_callbacks))
        {
            await callback.CompleteTraining(this, trainingSession.Epochs, _tensorTrainSet);
        }

        return Model;
    }
    
    public Prediction Predict(float[] input)
    {
        var processedInput = _tensorTrainSet.Process(input.ToTensor());
        return _optimizer switch
        {
            ISupervised supervised => supervised.Predict(processedInput),
            IUnsupervised unsupervised => unsupervised.Predict(processedInput),
            IEvolved evolved => evolved.Predict(processedInput),
            _ => throw new Exception("Cannot predict optimizer")
        };
    }

    public Prediction ProcessedPredict(float[] input) => _optimizer switch
    {
        ISupervised supervised => supervised.Predict(input.ToTensor()),
        IUnsupervised unsupervised => unsupervised.Predict(input.ToTensor()),
        IEvolved evolved => evolved.Predict(input.ToTensor()),
        _ => throw new Exception("Cannot predict optimizer")
    };

    public Validation ValidationScores() => Validate(_tensorTrainSet.TestingInputs);
    
    public Validation Validate(IEnumerable<float[]> inputs, IEnumerable<float[]> targets)
    {
        var batches = new TensorTrainSet(inputs, targets).BatchAll;
        return Validate(batches);
    }
    
    public OptimizerWrap Save() => new()
    {
        TensorOptions = _tensorTrainSet.TensorOptions,
        LossFunction = _loss,
        ModelWrap = Model switch
        {
            MultiLayerPerceptron perceptron => perceptron.Save(),
            RandomForest forest => forest.Save(),
            SupportVectorMachine vectorMachine => vectorMachine.Save(),
            KMeans means => means.Save(),
            Neat neat => neat.Save(),
            _ => throw new Exception("Cannot save optimizer")
        }
    };
    
    private TrainingSession GetTrainingSession() => _optimizer switch
    {
        IUnsupervised unsupervised => new UnsupervisedTrainingSession(unsupervised, _callbacks),
        ISupervised supervised => new SupervisedTrainingSession(supervised, _callbacks),
        _ => new EvolutionTrainingSession(_population, _callbacks),
    };

    private Validation Validate(List<Batch> batches)
    {
        var validator = new Validator(_loss switch
        {
            Loss.None => LossFunctionResolver.Get(_optimizer),
            _ => LossFunctionResolver.Get(_loss)
        });
        
        return _optimizer switch
        {
            ISupervised supervised => validator.Validate(supervised, batches),
            IUnsupervised unsupervised => validator.Validate(unsupervised, batches),
            _ => throw new Exception("Cannot validate model.")
        };
    }
    
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
                var forest = new RandomForest(modelWrap);
                return new Optimizer<T>(forest as T, trainSet, wrap.LossFunction, callbacks);
            }
            case ModelType.SVM:
            {
                var vectorMachine = new SupportVectorMachine(modelWrap);
                return new Optimizer<T>(vectorMachine as T, trainSet, wrap.LossFunction, callbacks);
            }
            case ModelType.KMeans:
            {
                var kMeans = new KMeans(modelWrap);
                return new Optimizer<T>(kMeans as T, trainSet, wrap.LossFunction, callbacks);
            }
            case ModelType.Neat:
            {
                var neat = new Neat(modelWrap);
                return new Optimizer<T>(neat as T, trainSet, Loss.None, callbacks);
            }
            default:
                return null;
        }
    }

}
