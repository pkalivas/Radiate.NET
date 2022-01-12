using Radiate.Tensors.Enums;

namespace Radiate.Records;

public record SpaceKernel(FeatureKernel FeatureKernel = FeatureKernel.Linear, float C = 1, float Gamma = -1);