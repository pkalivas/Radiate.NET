using Radiate.Activations;
using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Genomes.Forest.Info;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Genomes.Forest.Nodes;

public class NeuronTreeNode : Allele, ISeralTreeNode
{
    private readonly int[] _featureIndexes;
    private readonly float[] _weights;
    private readonly IActivationFunction _leafActivation;
    private IActivationFunction _activation;
    private float _outputCategory;
    private float _bias;

    public NeuronTreeNode(int index, int inputSize, float[] outputCategories, NeuronNodeInfo info) : base(index)
    {
        var (leafActivation, activations) = info;
        var activation = activations.ElementAt(Random.Next(0, activations.Count()));
        var indexCount = (int)Math.Max(1, Math.Ceiling(Random.NextDouble() * inputSize));
        
        _weights = new float[indexCount];
        _featureIndexes = new int[indexCount];
        _outputCategory = outputCategories[Random.Next(0, outputCategories.Length)];
        _bias = Random.NextSingle();
        _activation = ActivationFunctionFactory.Get(activation);
        _leafActivation = ActivationFunctionFactory.Get(leafActivation);
        
        for (var i = 0; i < indexCount; i++)
        {
            _weights[i] = Random.NextSingle() * 2f - 1f;
            _featureIndexes[i] = Random.Next(0, inputSize);
        }
    }

    public NeuronTreeNode(NeuronTreeNode node) : base(node.InnovationId)
    {
        _featureIndexes = node._featureIndexes.Select(val => val).ToArray();
        _outputCategory = node._outputCategory;
        _weights = node._weights.Select(val => val).ToArray();
        _bias = node._bias;
        _activation = ActivationFunctionFactory.Get(node._activation.ActivationType());
        _leafActivation = ActivationFunctionFactory.Get(node._leafActivation.ActivationType());
    }

    public NeuronTreeNode(NeuronTreeNodeWrap wrap)
    {
        _featureIndexes = wrap.FeatureIndexes;
        _outputCategory = wrap.OutputCategory;
        _weights = wrap.Weights;
        _bias = wrap.Bias;
        _activation = ActivationFunctionFactory.Get((Activation)wrap.Activation);
        _leafActivation = ActivationFunctionFactory.Get((Activation)wrap.LeafActivation);
    }

    public NeuronTreeNodeWrap Save() => new()
    {
        FeatureIndexes = _featureIndexes,
        OutputCategory = _outputCategory,
        Weights = _weights,
        Bias = _bias,
        Activation = (int)_activation.ActivationType(),
        LeafActivation = (int)_leafActivation.ActivationType()
    };
    
    public (int direction, Prediction prediction) Propagate(bool isLeaf, Tensor input, Prediction previousOutput)
    {
        var totalSum = _featureIndexes.Select((feature, index) => input[feature] * _weights[index]).Sum();
        var value = totalSum + _bias + (previousOutput?.Confidence ?? 0);
        var output = isLeaf ? _leafActivation.Activate(value) : _activation.Activate(value);
        var direction = ((int)_outputCategory) % 2 == 0f ? -1 : 1;
        var prediction = new Prediction(new[] { output }.ToTensor(), (int)_outputCategory, output);

        return (direction, prediction);
    }

    public void Mutate(ForestEnvironment environment)
    {
        var neuronSettings = environment.NeuronNodeSettings;
        if (Random.NextDouble() < neuronSettings.FeatureIndexMutateRate)
        {
            for (var i = 0; i < _featureIndexes.Length; i++)
            {
                if (Random.NextSingle() < neuronSettings.FeatureIndexMutateRate)
                {
                    _featureIndexes[i] = Random.Next(0, environment.InputSize);
                }
            }
        }
        
        if (Random.NextDouble() < neuronSettings.OutputCategoryMutateRate)
        {
            _outputCategory = environment.OutputCategories[Random.Next(0, environment.OutputCategories.Length)];
        }
        
        if (Random.NextDouble() < neuronSettings.WeightMutateRate)
        {
            var range = neuronSettings.WeightMovementRate;

            for (var i = 0; i < _weights.Length; i++)
            {
                var shouldEditWeight = Random.NextDouble() < neuronSettings.EditWeights;
                _weights[i] = shouldEditWeight ? Random.NextSingle() : _weights[i] + (Random.NextSingle() * (range * 2f) - range);   
            }

            var shouldEditBias = Random.NextDouble() < neuronSettings.EditWeights;
            _bias = shouldEditBias ? Random.NextSingle() : _bias + ((Random.NextSingle() * (range * 2f)) - range);
        }

        if (Random.NextDouble() < neuronSettings.ActivationMutateRate)
        {
            _activation = ActivationFunctionFactory.Get(neuronSettings.Activations.ElementAt(Random.Next(0, neuronSettings.Activations.Count())));
        }
    }

    public int InnovationNumber() => InnovationId;

    public float Weight() => _weights.Sum() + _bias;
}