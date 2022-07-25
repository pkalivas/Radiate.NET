using Radiate.Activations;

namespace Radiate.Optimizers.Evolution.Genomes.Forest.Info;

public interface INodeInfo { }

public record SeralForestInfo(int InputSize, float[] OutputCategories, int StartHeight, bool UseRecurrent, int NumTrees = 0);

public record NeuronNodeInfo(Activation LeafNodeActivation, IEnumerable<Activation> Activations) : INodeInfo;

public record OperatorNodeInfo() : INodeInfo;