using Radiate.Domain.Tensors;

namespace Radiate.Domain.Extensions;

public static class ArrayExtensions
{
    public static Tensor ToTensor(this float[] arr) => new(arr);

    public static Tensor ToTensor(this float[,] arr) => new(arr);

    public static Tensor ToTensor(this float[,,] arr) => new(arr);

    public static Tensor ToTensor(this IEnumerable<float> en) => en.ToArray().ToTensor();
}
