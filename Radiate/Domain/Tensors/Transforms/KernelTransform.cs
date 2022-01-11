using Radiate.Domain.Records;
using Radiate.Domain.Tensors.Enums;

namespace Radiate.Domain.Tensors.Transforms;

public class KernelTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train)
    {
        if (options.SpaceKernel is null)
        {
            return (trainTest, options);
        }
        
        var (featureKernel, c, gamma) = options.SpaceKernel;
        if (featureKernel is not FeatureKernel.Linear)
        {
            var (trainFeatures, _) = trainTest;
            var newFeatures = new Tensor[trainFeatures.Count];
            Parallel.For(0, newFeatures.Length, i =>
            {
                newFeatures[i] = featureKernel switch
                {
                    FeatureKernel.RBF => trainFeatures[i].RadialBasis(gamma),
                    FeatureKernel.Polynomial => trainFeatures[i].Polynomial(c),
                    _ => throw new Exception($"Kernel not implemented.")
                };
            });

            return (trainTest with { Features = newFeatures.ToList() }, options);
        }

        return (trainTest, options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options)
    {
        var kernel = options.SpaceKernel;
        if (kernel is null)
        {
            return value;
        }
        
        if (kernel.FeatureKernel is not FeatureKernel.Linear)
        {
            return kernel.FeatureKernel switch
            {
                FeatureKernel.Polynomial => value.Polynomial(kernel.C),
                FeatureKernel.RBF => value.RadialBasis(kernel.Gamma),
                _ => throw new Exception("Kernel not implemented")
            };
        }

        return value;
    }
    
}