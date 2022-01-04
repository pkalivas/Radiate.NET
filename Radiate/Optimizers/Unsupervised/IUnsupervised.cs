using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Unsupervised;

public interface IUnsupervised
{
    Prediction Predict(Tensor tensor, LossFunction lossFunction);
    Task Train(Batch batch, LossFunction lossFunction, Func<Epoch, bool> trainFunc);
}