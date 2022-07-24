﻿using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Forest.Info;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public class SeralForest : Allele, IGenome, IPredictionModel
{
    private readonly SeralTree[] _trees;
    private readonly SeralForestInfo _info;
    private Dictionary<int, float> _innovationWeightLookup;

    public SeralForest(SeralForestInfo info)
    {
        _info = info;
        _trees = Enumerable.Range(0, info.NumTrees).Select(_ => new SeralTree(info)).ToArray();
        _innovationWeightLookup = _trees
            .SelectMany(tree => tree.WeightLookup.Select(pair => (pair.Key, pair.Value)))
            .GroupBy(val => val.Key)
            .ToDictionary(key => key.Key, val => val.Sum(weight => weight.Value));
    }

    public SeralForest(SeralForest forest) : base(forest.InnovationId)
    {
        _info = forest._info with { };
        _trees = forest._trees.Select(tree => tree.CloneGenome<SeralTree>()).ToArray();
        _innovationWeightLookup = forest._innovationWeightLookup.ToDictionary(key => key.Key, val => val.Value);
    }

    public SeralForest(ModelWrap wrap)
    {
        var forest = wrap.SeralForestWrap;

        _info = new SeralForestInfo(forest.InputSize, Array.Empty<float>(), forest.MaxHeight, forest.NumTrees);
        _trees = forest.Trees.Select(tree => new SeralTree(tree)).ToArray();
        _innovationWeightLookup = _trees
            .SelectMany(tree => tree.WeightLookup.Select(pair => (pair.Key, pair.Value)))
            .GroupBy(val => val.Key)
            .ToDictionary(key => key.Key, val => val.Sum(weight => weight.Value));
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
        var child = CloneGenome<SeralForest>();
        var parentTwo = other as SeralForest;
        var treeEnv = environment as ForestEnvironment;

        for (var i = 0; i < _info.NumTrees; i++)
        {
            if (Random.NextDouble() < crossoverRate)
            {
                child._trees[i] = child._trees[i].Crossover(parentTwo._trees[i], treeEnv, crossoverRate);
            }
            else
            {
                var index = Random.Next(0, _info.NumTrees);
                child._trees[i] = child._trees[i].Crossover(child._trees[index], treeEnv, crossoverRate);
            }
        }
        
        child._innovationWeightLookup = _trees
            .SelectMany(tree => tree.WeightLookup.Select(pair => (pair.Key, pair.Value)))
            .GroupBy(val => val.Key)
            .ToDictionary(key => key.Key, val => val.Sum(weight => weight.Value));
        
        return child as T;
    }

    public async Task<double> Distance<T>(T other, PopulationControl populationControl)
    {
        var parentTwo = other as SeralForest;
        return await DistanceCalculator.Distance(_innovationWeightLookup, parentTwo._innovationWeightLookup, populationControl);
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