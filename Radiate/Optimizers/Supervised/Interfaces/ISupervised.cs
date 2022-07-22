using Radiate.IO.Wraps;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Supervised.Interfaces;

public interface ISupervised : IOptimizerModel, IPredictionModel
{
    List<Step> Step(Tensor[] features, Tensor[] targets);
    void Update(List<Cost> errors, int epochCount);
    ModelWrap Save();
}