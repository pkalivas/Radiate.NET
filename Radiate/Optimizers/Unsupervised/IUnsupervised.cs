using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Unsupervised;

public interface IUnsupervised
{
    Prediction Predict(Tensor tensor);
    void Train(Tensor[] data, Func<Epoch, bool> trainFunc);
    UnsupervisedWrap Save();
}