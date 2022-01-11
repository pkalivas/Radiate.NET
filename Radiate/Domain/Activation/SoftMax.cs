using Radiate.Domain.Extensions;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Activation;

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

    public Tensor Deactivate(Tensor values)
    {
        // var diagMatrix = values.Diag();
        // var tiledMatrix = values.Tile();
        // var transposedMatrix = values.T();
        //
        // var result = new float[values.Count()];
        // for (var i = 0; i < values.Count(); i++)
        // {
        //     for (var j = 0; j < values.Count(); j++)
        //     {
        //         result[i] += diagMatrix[j, i] - (tiledMatrix[j, i] * transposedMatrix[j, i]);
        //     }
        // }
        //
        //
        return values.Select(val => val switch
        {
            > MaxClipValue => MaxClipValue,
            < MinClipValue => MinClipValue,
            var x and >= MinClipValue and <= MaxClipValue => x,
            var x => throw new Exception($"Failed to activate Softmax {x}")
        }).ToArray().ToTensor();
    }

    public float Activate(float value) => throw new Exception($"Softmax of single value is not real.");

    public float Deactivate(float value) => throw new Exception($"Cannot take dSoftmax of single value");

    public Activation ActivationType() => Activation.SoftMax;

}
