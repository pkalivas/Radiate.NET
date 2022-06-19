using Radiate.Callbacks.Interfaces;
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

public interface IOptimizerModel { }

public class Optimizer
{
    private readonly IOptimizerModel _optimizer;
    private readonly Loss _loss;
    private readonly TensorTrainSet _tensorTrainSet;
    private readonly TrainingSession _trainingSession;
    
    public Optimizer(IPopulation population, TensorTrainSet tensorTrainSet = null)
    {
        _tensorTrainSet = tensorTrainSet?.Compile();
        _trainingSession = new EvolutionTrainingSession(population, new List<ITrainingCallback>());
    }

    public Optimizer(OptimizerWrap wrap) 
        : this(Load(wrap), new TensorTrainSet(wrap.TensorOptions), wrap.LossFunction) { }
    
    public Optimizer(IOptimizerModel optimizer, TensorTrainSet tensorTrainSet, IEnumerable<ITrainingCallback> callbacks)
        : this(optimizer, tensorTrainSet, Loss.None, callbacks) { }

    public Optimizer(IOptimizerModel optimizer, Loss loss = Loss.None, IEnumerable<ITrainingCallback> callbacks = null) 
        : this(optimizer, null, loss, callbacks) { }

    public Optimizer(IOptimizerModel optimizer, TensorTrainSet tensorTrainSet, Loss loss = Loss.None, IEnumerable<ITrainingCallback> callbacks = null)
    {
        _optimizer = optimizer;
        _tensorTrainSet = tensorTrainSet;
        _loss = loss;
        _trainingSession = _optimizer switch
        {
            IUnsupervised unsupervised => new UnsupervisedTrainingSession(unsupervised, callbacks),
            ISupervised supervised => new SupervisedTrainingSession(supervised, callbacks),
            _ => throw new Exception($"Cannot create training session for model.")
        };
        
        Model = _optimizer;
    }
    
    private IOptimizerModel Model { get; set; }
    
    private LossFunction LossFunction => _loss switch
    {
        Loss.None => LossFunctionResolver.Get(_optimizer),
        _ => LossFunctionResolver.Get(_loss)
    };

    public async Task<T> Train<T>() where T : class, IOptimizerModel => await Train<T>(_ => Task.FromResult(true));

    public async Task<T> Train<T>(Func<Epoch, bool> trainFunc) where T : class, IOptimizerModel =>
        await Train<T>(epoch => Task.Run(() => trainFunc(epoch)));

    private async Task<T> Train<T>(Func<Epoch, Task<bool>> trainFunc) where T : class, IOptimizerModel
    {
        Model = await _trainingSession.Train(_tensorTrainSet, trainFunc, LossFunction);
        
        await _trainingSession.CompleteTraining(this, _tensorTrainSet);
        
        return Model as T;
    }
    
    public Prediction Predict(float[] input)
    {
        var processedInput = _tensorTrainSet.Process(input.ToTensor());
        return Model switch
        {
            ISupervised supervised => supervised.Predict(processedInput),
            IUnsupervised unsupervised => unsupervised.Predict(processedInput),
            IEvolved evolved => evolved.Predict(processedInput),
            _ => throw new Exception("Cannot predict optimizer")
        };
    }

    public Prediction ProcessedPredict(float[] input) => Model switch
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

    private Validation Validate(List<Batch> batches)
    {
        var validator = new Validator(LossFunction);
        
        return Model switch
        {
            ISupervised supervised => validator.Validate(supervised, batches),
            IUnsupervised unsupervised => validator.Validate(unsupervised, batches),
            _ => throw new Exception("Cannot validate model.")
        };
    }

    private static IOptimizerModel Load(OptimizerWrap wrap) => wrap.ModelWrap.ModelType switch
    {
        ModelType.MultiLayerPerceptron => new MultiLayerPerceptron(wrap.ModelWrap),
        ModelType.RandomForest => new RandomForest(wrap.ModelWrap),
        ModelType.SVM => new SupportVectorMachine(wrap.ModelWrap),
        ModelType.KMeans => new KMeans(wrap.ModelWrap),
        ModelType.Neat => new Neat(wrap.ModelWrap),
        _ => null
    };
}
