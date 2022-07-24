using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Forest.Info;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public class OperatorTreeNode : Allele, ISeralTreeNode
{
    private int _splitIndex;
    private float _outputCategory;
    private float _splitValue;
    private Operator _operator;

    public OperatorTreeNode(int index, SeralForestInfo info) : base(index)
    {
        var (_, _, inputSize, outputCategories, _, _, _) = info;
        _splitIndex = Random.Next(0, inputSize);
        _outputCategory = outputCategories[Random.Next(0, outputCategories.Length)];
        _splitValue = (Random.NextSingle() * 2) - 1;
        _operator = (Operator)Random.Next(0, 3);
    }

    public OperatorTreeNode(OperatorTreeNode node) : base(node.InnovationId)
    {
        _splitIndex = node._splitIndex;
        _splitValue = node._splitValue;
        _outputCategory = node._outputCategory;
        _operator = node._operator;
    }

    public OperatorTreeNode(OperatorTreeNodeWrap wrap)
    {
        _splitIndex = wrap.SplitIndex;
        _splitValue = wrap.SplitValue;
        _outputCategory = wrap.OutputCategory;
        _operator = (Operator)wrap.Operator;
    }

    public OperatorTreeNodeWrap Save() => new()
    {
        SplitIndex = _splitIndex,
        SplitValue = _splitValue,
        OutputCategory = _outputCategory,
        Operator = (int)_operator
    };
    
    public Prediction Predict(Tensor input) => new(new[] { _splitValue }.ToTensor(), (int)_outputCategory, _outputCategory);

    public NodePropagationDirection GetDirection(Tensor input)
    {
        var result = _operator switch
        {
            Operator.EqualTo => input[_splitIndex] == _splitValue,
            Operator.GreaterThan => input[_splitIndex] > _splitValue,
            Operator.LessThan => input[_splitIndex] < _splitValue,
            _ => throw new Exception("Operator not implemented")
        };

        return result ? NodePropagationDirection.Left : NodePropagationDirection.Right;
    }

    public void Mutate(ForestEnvironment environment)
    {
        var operatorEnvironment = environment.OperatorNodeEnvironment;
        if (Random.NextDouble() < operatorEnvironment.SplitValueMutateRate)
        {
            _splitValue += ((float)Random.NextDouble() * 2) - 1;
        }
        
        if (Random.NextDouble() < operatorEnvironment.SplitIndexMutateRate)
        {
            _splitIndex = Random.Next(0, environment.InputSize);
        }
        
        if (Random.NextDouble() < operatorEnvironment.OutputCategoryMutateRate)
        {
            _outputCategory = environment.OutputCategories[Random.Next(0, environment.OutputCategories.Length)];
        }
        
        if (Random.NextDouble() < operatorEnvironment.OperatorMutateRate)
        {
            _operator = (Operator)Random.Next(0, 3);
        }
    }
    
    public int InnovationNumber() => InnovationId;

    public float Weight() => _splitValue;
}