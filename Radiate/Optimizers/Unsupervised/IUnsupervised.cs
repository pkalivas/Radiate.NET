using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Unsupervised;

public interface IUnsupervised
{
    Prediction Predict(Tensor tensor);
    float Step(Tensor[] data, int epochCount);
    void Update();
    UnsupervisedWrap Save();
}