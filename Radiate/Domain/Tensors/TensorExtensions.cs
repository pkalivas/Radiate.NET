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

    public static string ConfusionMatrix(this Tensor one)
    {
        var shape = one.Shape;
        var result = $"{"",13}" + string.Join($"| {"",5}", Enumerable.Range(0, shape.Width)) + "|\n";
        var sep = string.Join("-", Enumerable.Range(0, result.Length).Select(_ => "")) + "\n";
        result += sep;
        for (var i = 0; i < shape.Height; i++)
        {
            result += $"{i,5} | ";
            for (var j = 0; j < shape.Width; j++)
            {
                var val = one[i, j];
                var dis = $"{val,5}";
                result += $"{dis}   ";
            }

            result += "\n";
        }

        return result;
    }
}