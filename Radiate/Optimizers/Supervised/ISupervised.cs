using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised;

public interface ISupervised
{
    Prediction Predict(Tensor input);
    void Train(List<Batch> data, LossFunction lossFunction, Func<Epoch, bool> trainFunc);
}