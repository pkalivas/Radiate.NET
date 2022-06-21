using Radiate.Callbacks.Interfaces;
using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Losses;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Environment;
using Radiate.Optimizers.Evolution.Forest;
using Radiate.Optimizers.Evolution.Neat;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.SVM;
using Radiate.Optimizers.Unsupervised;
using Radiate.Optimizers.Unsupervised.Clustering;
using Radiate.Records;
using Radiate.Tensors;
using Radiate.TrainingSessions;

namespace Radiate.Optimizers;

public interface IOptimizerModel { }

public class Optimizer
{
    public IOptimizerModel Model { get; private set; }

    private readonly ILossFunction _lossFunction;
    private readonly TensorTrainSet _tensorTrainSet;
    private readonly TrainingSession _trainingSession;
    
    public Optimizer(IPopulation population, TensorTrainSet tensorTrainSet = null, IEnumerable<ITrainingCallback> callbacks = null)
    {
        _tensorTrainSet = tensorTrainSet?.Compile();
        _lossFunction = new Difference();
        _trainingSession = new EvolutionTrainingSession(population, callbacks);
    }

    public Optimizer(OptimizerWrap wrap) 
        : this(Load(wrap), new TensorTrainSet(wrap.TensorOptions), wrap.LossFunction) { }
    
    public Optimizer(IOptimizerModel optimizer, TensorTrainSet tensorTrainSet, IEnumerable<ITrainingCallback> callbacks)
        : this(optimizer, tensorTrainSet, Loss.Difference, callbacks) { }

    public Optimizer(IOptimizerModel optimizer, TensorTrainSet tensorTrainSet, Loss loss = Loss.Difference, IEnumerable<ITrainingCallback> callbacks = null)
    {
        Model = optimizer;

        _tensorTrainSet = tensorTrainSet;
        _lossFunction = LossFunctionResolver.Get(loss);
        _trainingSession = optimizer switch
        {
            IUnsupervised unsupervised => new UnsupervisedTrainingSession(unsupervised, callbacks),
            ISupervised supervised => new SupervisedTrainingSession(supervised, callbacks),
            IGenome genome => new EvolutionTrainingSession(genome, callbacks),
            _ => throw new Exception("Cannot create training session")
        };
    }
    
    public async Task<T> Train<T>() where T : class, IOptimizerModel => await Train<T>(_ => Task.FromResult(true));

    public async Task<T> Train<T>(Func<Epoch, bool> trainFunc) where T : class, IOptimizerModel =>
        await Train<T>(epoch => Task.Run(() => trainFunc(epoch)));

    public async Task<T> Train<T>(Func<Epoch, Task<bool>> trainFunc) where T : class, IOptimizerModel
    {
        Model = await _trainingSession.Train(_tensorTrainSet, trainFunc, _lossFunction.Calculate);
        
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
        LossFunction = _lossFunction.LossType(),
        ModelWrap = Model switch
        {
            MultiLayerPerceptron perceptron => perceptron.Save(),
            RandomForest forest => forest.Save(),
            SupportVectorMachine vectorMachine => vectorMachine.Save(),
            KMeans means => means.Save(),
            Neat neat => neat.Save(),
            SeralTree tree => tree.Save(),
            SeralForest forest => forest.Save(),
            _ => throw new Exception("Cannot save optimizer")
        }
    };

    private Validation Validate(List<Batch> batches)
    {
        var validator = new Validator(_lossFunction.Calculate);
        return validator.Validate(Model, batches);
    }

    private static IOptimizerModel Load(OptimizerWrap wrap) => wrap.ModelWrap.ModelType switch
    {
        ModelType.MultiLayerPerceptron => new MultiLayerPerceptron(wrap.ModelWrap),
        ModelType.RandomForest => new RandomForest(wrap.ModelWrap),
        ModelType.SVM => new SupportVectorMachine(wrap.ModelWrap),
        ModelType.KMeans => new KMeans(wrap.ModelWrap),
        ModelType.Neat => new Neat(wrap.ModelWrap),
        ModelType.SeralTree => new SeralTree(wrap.ModelWrap),
        ModelType.SeralForest => new SeralForest(wrap.ModelWrap),
        _ => null
    };
}
