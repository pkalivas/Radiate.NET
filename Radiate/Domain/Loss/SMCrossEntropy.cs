using System.Numerics;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class SMCrossEntropy : ILossFunction
{
    public Cost Calculate(float[] output, float[] target)
    {
        var correctIndex = target.ToList().IndexOf(target.Max());
        var result = Tensor.Fill(new float[output.Length].ToTensor().Shape, 1f);
        var error = output[correctIndex];
        var loss = error > 0f 
                ? -(float)Math.Log(error) 
                : -(float) Complex.Log(error).Imaginary;
        
        var errors = output.Zip(target)
            .Select(pair => -pair.Second * (float)Math.Log(pair.First))
            .ToArray();
        result[correctIndex] = -1 / errors[correctIndex];

        return new Cost(result.Read1D(), loss);
    }
}