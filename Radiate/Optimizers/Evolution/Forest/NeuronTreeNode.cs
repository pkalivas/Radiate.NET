using Radiate.Activations;
using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Forest.Info;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public class NeuronTreeNode : Allele, ISeralTreeNode
{
    private int _splitIndex;
    private float _outputCategory;
    private float _weight;
    private float _bias;
    private float _previousOutput;
    private bool _recurrent;
    private IActivationFunction _activation;

    public NeuronTreeNode(int index, SeralForestInfo info) : base(index)
    {
        var (_, activations, inputSize, outputCategories, _, useRecurrent, _) = info;
        _splitIndex = Random.Next(0, inputSize);
        _outputCategory = outputCategories[Random.Next(0, outputCategories.Length)];
        _weight = (Random.NextSingle() * 2) - 1;
        _bias = Random.NextSingle();
        _previousOutput = 0f;
        _recurrent = useRecurrent;
        _activation = ActivationFunctionFactory.Get(activations.ElementAt(Random.Next(0, activations.Count())));
    }

    public NeuronTreeNode(NeuronTreeNode node) : base(node.InnovationId)
    {
        _splitIndex = node._splitIndex;
        _outputCategory = node._outputCategory;
        _weight = node._weight;
        _bias = node._bias;
        _recurrent = node._recurrent;
        _previousOutput = 0f;
        _activation = ActivationFunctionFactory.Get(node._activation.ActivationType());
    }

    public NeuronTreeNode(NeuronTreeNodeWrap wrap)
    {
        _splitIndex = wrap.SplitIndex;
        _outputCategory = wrap.OutputCategory;
        _weight = wrap.Weight;
        _bias = wrap.Bias;
        _recurrent = wrap.Recurrent;
        _previousOutput = 0f;
        _activation = ActivationFunctionFactory.Get((Activation)wrap.Activation);
    }

    public NeuronTreeNodeWrap Save() => new()
    {
        SplitIndex = _splitIndex,
        OutputCategory = _outputCategory,
        Weight = _weight,
        Bias = _bias,
        Recurrent = _recurrent,
        Activation = (int)_activation.ActivationType()
    };

    public Prediction Predict(Tensor input)
    {
        var value = input[_splitIndex] * _weight + _bias;
        var output = _activation.Activate(value + _previousOutput);

        if (_recurrent)
        {
            _previousOutput = output;
        }
        
        return new Prediction(new[] { output }.ToTensor(), (int)_outputCategory, output);
    }

    public NodePropagationDirection GetDirection(Tensor input)
    {
        var value = input[_splitIndex] * _weight + _bias;
        var output = _activation.Activate(value + _previousOutput);

        if (_recurrent)
        {
            _previousOutput = output;
        }
        
        return _previousOutput > 0.5 ? NodePropagationDirection.Left : NodePropagationDirection.Right;
    }

    public void Mutate(ForestEnvironment environment)
    {
        var neuronSettings = environment.NeuronNodeEnvironment;
        if (Random.NextDouble() < neuronSettings.SplitIndexMutateRate)
        {
            _splitIndex = Random.Next(0, environment.InputSize);
        }
        
        if (Random.NextDouble() < neuronSettings.OutputCategoryMutateRate)
        {
            _outputCategory = environment.OutputCategories[Random.Next(0, environment.OutputCategories.Length)];
        }
        
        if (Random.NextDouble() < neuronSettings.WeightMutateRate)
        {
            var range = neuronSettings.WeightMovementRate;
            var shouldEditWeight = Random.NextDouble() < neuronSettings.EditWeights;
            _weight = shouldEditWeight ? Random.NextSingle() : _weight * (Random.NextSingle() * range - range);

            var shouldEditBias = Random.NextDouble() < neuronSettings.EditWeights;
            _bias = shouldEditBias ? Random.NextSingle() : _bias * (Random.NextSingle() * range - range);
        }

        if (Random.NextDouble() < neuronSettings.ActivationMutateRate)
        {
            _activation = ActivationFunctionFactory.Get(neuronSettings.Activations.ElementAt(Random.Next(0, neuronSettings.Activations.Count())));
        }
    }

    public int InnovationNumber() => InnovationId;

    public float Weight() => _weight + _bias;
}