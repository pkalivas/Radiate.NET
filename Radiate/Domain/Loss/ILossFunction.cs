using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public interface ILossFunction
{
    Cost Calculate(Tensor output, Tensor target);
}