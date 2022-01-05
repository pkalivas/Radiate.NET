namespace Radiate.Domain.Tensors;

public static class TensorExtensions
{
    public static string ToImageString(this Tensor one, float scale = .5f)
    {
        var shape = one.Shape;
        var result = "";
        for (var i = 0; i < shape.Height; i++)
        {
            for (var j = 0; j < shape.Width; j++)
            {
                for (var k = 0; k < shape.Depth; k++)
                {
                    var value = one[i, j, k];
                    result += value > scale ? "*" : " ";
                }
            }

            result += "\n";
        }

        return result;
    }
}