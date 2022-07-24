using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public interface ISeralTreeNode
{
    public Prediction Predict(Tensor input);
    public NodePropagationDirection GetDirection(Tensor input);
    public void Mutate(ForestEnvironment environment);
    public int InnovationNumber();
    public float Weight();
}