using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution;

public interface IEvolved
{
    public Prediction Predict(Tensor input);
}