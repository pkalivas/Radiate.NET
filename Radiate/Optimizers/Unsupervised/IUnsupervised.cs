using Radiate.IO.Wraps;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Unsupervised;

public interface IUnsupervised : IOptimizerModel
{
    Prediction Predict(Tensor tensor);
    float Step(Tensor[] data, int epochCount);
    ModelWrap Save();
}