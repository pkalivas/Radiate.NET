using System.Threading.Tasks;
using Radiate.Domain.Gradients;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised
{
    public interface IOptimizer
    {
        Tensor Predict(Tensor  inputs);
        Tensor  PassForward(Tensor  inputs);
        void PassBackward(Tensor  errors, int epoch);
        Task Update(GradientInfo gradient, int epoch);
    }
}