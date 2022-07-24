using Radiate.Callbacks.Interfaces;
using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Losses;
using Radiate.Optimizers.Evolution.Forest;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Optimizers.Evolution.Neat;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Interfaces;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.SVM;
using Radiate.Optimizers.Unsupervised.Clustering;
using Radiate.Optimizers.Unsupervised.Interfaces;
using Radiate.Records;
using Radiate.Tensors;
using Radiate.TrainingSessions;

namespace Radiate.Optimizers;

public interface IOptimizerModel { }

public interface IPredictionModel
{
    public Prediction Predict(Tensor input);
}

public class Optimizer
{
    public IOptimizerModel Model { get; private set; }

    private readonly ILossFunction _lossFunction;
    private readonly TensorTrainSet _tensorTrainSet;
    private readonly TrainingSession _trainingSession;

    public Optimizer(OptimizerWrap wrap) 
        : this(Load(wrap), new TensorTrainSet(wrap.TensorOptions), wrap.LossFunction) { }
    
    public Optimizer(IOptimizerModel optimizer, IEnumerable<ITrainingCallback> callbacks = null) 
        : this(optimizer, new(), callbacks) { }
    
    public Optimizer(IOptimizerModel optimizer, TensorTrainSet tensorTrainSet, IEnumerable<ITrainingCallback> callbacks)
        : this(optimizer, tensorTrainSet, Loss.Difference, callbacks) { }

    public Optimizer(IOptimizerModel optimizer, TensorTrainSet tensorTrainSet, Loss loss = Loss.Difference, IEnumerable<ITrainingCallback> callbacks = null)
    {
        Model = optimizer;

        _tensorTrainSet = tensorTrainSet?.Compile();
        _lossFunction = LossFunctionResolver.Get(loss);
        _trainingSession = optimizer switch
        {
            IUnsupervised unsupervised => new UnsupervisedTrainingSession(unsupervised, callbacks),
            ISupervised supervised => new SupervisedTrainingSession(supervised, callbacks),
            IGenome genome => new EvolutionTrainingSession(genome, callbacks),
            IPopulation population => new EvolutionTrainingSession(population, callbacks),
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

        if (Model is IPredictionModel predictionModel)
        {
            return predictionModel.Predict(processedInput);
        }
        
        throw new Exception("Model is not of type IPredictionModel.");
    }

    public Prediction ProcessedPredict(float[] input)
    {
        if (Model is IPredictionModel predictionModel)
        {
            return predictionModel.Predict(input.ToTensor());
        }

        throw new Exception("Model is not of type IPredictionModel.");
    }

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

        if (Model is IPredictionModel predictionModel)
        {
            return validator.Validate(predictionModel, batches);
        }

        throw new Exception("Cannot validate model not of type IPredictionModel.");
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
