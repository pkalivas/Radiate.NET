using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Services;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers;

public interface IPredictor
{
    Prediction Predict(Tensor input);
}