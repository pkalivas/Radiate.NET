using Radiate.IO.Wraps;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Supervised;

public interface ISupervised
{
    Prediction Predict(Tensor input);
    List<Step> Step(Tensor[] features, Tensor[] targets);
    void Update(List<Cost> errors, int epochCount);
    ModelWrap Save();
}