using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Forest.Info;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public class SeralForest : IGenome, IEvolved, IOptimizerModel
{
    private readonly SeralTree[] _trees;
    private readonly SeralForestInfo _info;
    
    public SeralForest(SeralForestInfo info)
    {
        _info = info;
        _trees = Enumerable.Range(0, info.NumTrees)
            .Select(_ => new SeralTree(info))
            .ToArray();
    }

    public SeralForest(SeralForest forest)
    {
        _info = forest._info with { };
        _trees = forest._trees.Select(tree => tree.CloneGenome<SeralTree>()).ToArray();
    }

    public SeralForest(ModelWrap wrap)
    {
        var forest = wrap.SeralForestWrap;

        _info = new SeralForestInfo(forest.InputSize, new float[0] {}, forest.MaxHeight, forest.NumTrees);
        _trees = forest.Trees.Select(tree => new SeralTree(tree)).ToArray();
    }

    public ModelWrap Save() => new()
    {
        ModelType = ModelType.SeralForest,
        SeralForestWrap = new()
        {
            InputSize = _info.InputSize,
            MaxHeight = _info.MaxHeight,
            NumTrees = _info.NumTrees,
            Trees = _trees.Select(tree => tree.Save()).ToList()
        }
    };

    public T Crossover<T, TE>(T other, TE environment, double crossoverRate) where T : class, IGenome where TE : EvolutionEnvironment
    {
        var random = new Random();

        var child = CloneGenome<SeralForest>();
        var parentTwo = other as SeralForest;
        var treeEnv = environment as ForestEnvironment;

        for (var i = 0; i < _info.NumTrees; i++)
        {
            if (random.NextDouble() < crossoverRate)
            {
                child._trees[i] = child._trees[i].Crossover(parentTwo._trees[i], treeEnv, crossoverRate);
            }
            else
            {
                var index = random.Next(0, _info.NumTrees);
                child._trees[i] = child._trees[i].Crossover(child._trees[index], treeEnv, crossoverRate);
            }
        }
        
        return child as T;
    }

    public async Task<double> Distance<T, TE>(T other, TE environment)
    {
        var parentTwo = other as SeralForest;
        var result = 0.0;
        foreach (var (one, two) in _trees.Zip(parentTwo._trees))
        {
            result += await one.Distance(two, environment);
        }

        return result;
    }

    public T CloneGenome<T>() where T : class => new SeralForest(this) as T;

    public void ResetGenome()
    {
        foreach (var tree in _trees)
        {
            tree.ResetGenome();
        }
    }

    public Prediction Predict(Tensor input)
    {
        var predictions = _trees.Select(tree => tree.Predict(input)).ToList();
        var confidence = predictions.Select(pred => pred.Confidence).Average();
        var classification = predictions
            .Select(pred => pred.Classification)
            .GroupBy(val => val)
            .OrderByDescending(val => val.Count())
            .First().Key;

        return new Prediction(new[] { confidence }.ToTensor(), classification, confidence);
    }
}