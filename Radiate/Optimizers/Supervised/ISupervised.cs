using Radiate.Domain.Loss;
using Radiate.Domain.Records;

namespace Radiate.Optimizers.Supervised;

public interface ISupervised
{
    Task Train<T>(List<Batch<T>> data, LossFunction lossFunction, Func<Epoch, bool> trainFunc);
}