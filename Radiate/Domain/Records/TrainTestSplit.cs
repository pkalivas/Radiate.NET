using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record TrainTestSplit(List<Tensor> TrainFeatures, List<Tensor> TrainTargets, List<Tensor> TestFeatures, List<Tensor> TestTargets);