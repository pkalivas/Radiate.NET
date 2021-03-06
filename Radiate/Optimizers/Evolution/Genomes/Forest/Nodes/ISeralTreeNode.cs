using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Genomes.Forest.Nodes;

public interface ISeralTreeNode
{
    public (int direction, Prediction prediction) Propagate(bool isLeaf, Tensor input, Prediction previousOutput);
    public void Mutate(ForestEnvironment environment);
    public int InnovationNumber();
    public float Weight();
}