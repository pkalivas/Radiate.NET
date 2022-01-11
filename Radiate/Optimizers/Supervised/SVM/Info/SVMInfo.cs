using Radiate.Domain.Records;

namespace Radiate.Optimizers.Supervised.SVM.Info;

public record SVMInfo(Shape FeatureShape, int NumClasses, float Lambda = 1e-5f);