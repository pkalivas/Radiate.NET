using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Losses;

public interface ILossFunction
{
    Loss LossType();
    Cost Calculate(Tensor output, Tensor target);
}