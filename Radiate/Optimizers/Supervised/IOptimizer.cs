using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised;

public interface IOptimizer
{
    Prediction Predict(Tensor input);
    Task Train(List<Batch> batches, LossFunction lossFunction, Func<Epoch, bool> trainFunc);
    OptimizerWrap Save();
}
