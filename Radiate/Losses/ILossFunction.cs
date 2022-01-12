using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Losses;

public interface ILossFunction
{
    Cost Calculate(Tensor output, Tensor target);
}