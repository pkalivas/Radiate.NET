using Radiate.Domain.Tensors.Enums;

namespace Radiate.Domain.Records;

public record SpaceKernel(FeatureKernel FeatureKernel = FeatureKernel.Linear, float C = 1, float Gamma = -1);