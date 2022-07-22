using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IEvolved
{
    public Prediction Predict(Tensor input);
}