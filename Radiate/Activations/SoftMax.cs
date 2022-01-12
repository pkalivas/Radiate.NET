using Radiate.Extensions;
using Radiate.Tensors;

namespace Radiate.Activations;

public class SoftMax : IActivationFunction
{
    private const float MaxClipValue = 200;
    private const float MinClipValue = -200;
    
    public Tensor Activate(Tensor values)
    {
        var expSum = values.Sum(val => (float)Math.Exp(val));
        return values
            .Select(val => (float)Math.Exp(val) / expSum)
            .ToTensor();
    }

    public Tensor Deactivate(Tensor values) =>
        values.Select(val => val switch
        {
            > MaxClipValue => MaxClipValue,
            < MinClipValue => MinClipValue,
            var x and >= MinClipValue and <= MaxClipValue => x,
            var x => throw new Exception($"Failed to activate Softmax {x}")
        }).ToTensor();

    public float Activate(float value) => throw new Exception($"Softmax of single value is not real.");

    public float Deactivate(float value) => throw new Exception($"Cannot take dSoftmax of single value");

    public Activation ActivationType() => Activation.SoftMax;

}
