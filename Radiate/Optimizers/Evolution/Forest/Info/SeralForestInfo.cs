using Radiate.Activations;

namespace Radiate.Optimizers.Evolution.Forest.Info;

public interface INodeInfo { }

public record SeralForestInfo(int InputSize, float[] OutputCategories, int StartHeight, bool UseRecurrent, int NumTrees = 0);

public record NeuronNodeInfo(Activation LeafNodeActivation, IEnumerable<Activation> Activations, int FeatureIndexCount) : INodeInfo;

public record OperatorNodeInfo() : INodeInfo;