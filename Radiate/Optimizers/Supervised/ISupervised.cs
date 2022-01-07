using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised;

public interface ISupervised
{
    Prediction Predict(Tensor input);
    List<(Prediction prediction, Tensor target)> Step(Tensor[] features, Tensor[] targets);
    void Update(List<Cost> errors, int epochCount);
    SupervisedWrap Save();
}