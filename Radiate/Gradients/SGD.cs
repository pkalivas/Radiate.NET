using Radiate.Tensors;

namespace Radiate.Gradients;

public class SGD : IGradient
{
    private readonly float _learningRate;
    
    public SGD(float learningRate)
    {
        _learningRate = learningRate;
    }

    public Tensor Calculate(Tensor gradients, int epoch)
    {
        gradients.Mul(_learningRate);
        return gradients;
    }
}