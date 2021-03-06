using Radiate.Tensors;

namespace Radiate.Activations;

public class Sigmoid : IActivationFunction
{
    private const float MinClipValue = -5f;
    private const float MaxClipValue = 5f;
    
    public Tensor Activate(Tensor values) => Tensor.Apply(values, Calc);

    public Tensor Deactivate(Tensor values) => Tensor.Apply(values, DCalc);

    public float Activate(float value) => Calc(value);

    public float Deactivate(float value) => DCalc(value);

    public Activation ActivationType() => Activation.Sigmoid;


    private static float Calc(float val) => 1f / (1f + (float)Math.Exp(-val));

    private static float DCalc(float val) => val * (1 - val) switch
    {
        > MaxClipValue => MaxClipValue,
        < MinClipValue => MinClipValue,
        var x and >= MinClipValue and <= MaxClipValue => x,
        var x => throw new Exception($"Failed to activate Sigmoid {x}")
    };
}