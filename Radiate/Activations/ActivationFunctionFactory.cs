
namespace Radiate.Activations;

public static class ActivationFunctionFactory
{
    public static IActivationFunction Get(Activation activation) => activation switch
    {
        Activation.Sigmoid => new Sigmoid(),
        Activation.ReLU => new ReLU(),
        Activation.Linear => new Linear(),
        Activation.Tanh => new Tanh(),
        Activation.SoftMax => new SoftMax(),
        Activation.ExpSigmoid => new ExpSigmoid(),
        _ => throw new Exception($"Activation {activation} is not implemented.")
    };
}