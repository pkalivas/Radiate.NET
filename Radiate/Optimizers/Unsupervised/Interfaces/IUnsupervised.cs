using Radiate.IO.Wraps;
using Radiate.Tensors;

namespace Radiate.Optimizers.Unsupervised.Interfaces;

public interface IUnsupervised : IOptimizerModel, IPredictionModel
{
    float Step(Tensor[] data, int epochCount);
    ModelWrap Save();
}