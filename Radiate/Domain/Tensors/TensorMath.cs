namespace Radiate.Domain.Tensors;

public static class TensorMath
{
    public static void AddInPlace(this Tensor one, Tensor two)
    {
        var (oHeight, oWidth, oDepth) = one.Shape;
        
        for (var j = 0; j < oHeight; j++)
            if (oWidth > 0)
                for (var k = 0; k < oWidth; k++)
                    if (oDepth > 0)
                        for (var l = 0; l < oDepth; l++)
                            one[j, k, l] += two[j, k, l];
                    else
                        one[j, k] += two[j, k];
            else
                one[j] += two[j];
    }
    
    public static void MultiplyInPlace(this Tensor one, float two)
    {
        var (oHeight, oWidth, oDepth) = one.Shape;
        
        for (var j = 0; j < oHeight; j++)
            if (oWidth > 0)
                for (var k = 0; k < oWidth; k++)
                    if (oDepth > 0)
                        for (var l = 0; l < oDepth; l++)
                            one[j, k, l] *= two;
                    else
                        one[j, k] *= two;
            else
                one[j] *= two;
    }
    
    public static float Dot(Tensor one, Tensor two)
    {
        var (oHeight, oWidth, oDepth) = one.Shape;
        var (tHeight, tWidth, tDepth) = two.Shape;
        var oDim = one.GetDimension();
        var tDim = two.GetDimension();
        
        if (oHeight != tHeight || oWidth != tWidth || oDepth != tDepth)
        {
            throw new Exception($"Cannot dot product two unlike matrices");
        }

        var result = 0.0f;
        
        if (oDim == 1 && tDim == 1)
        {
            Tensor.TensorLoop(i => result += one[i] * two[i], oHeight);
            return result;
        }

        if (oDim == 2 && tDim == 2)
        {
            Tensor.TensorLoop((i, j) => result += one[i, j] * two[i, j], oHeight, oWidth);
            return result;
        }

        if (oDim == 3 && tDim == 3)
        {
            Tensor.TensorLoop((i, j, k) => result += one[i, j, k] * two[i, j, k], oHeight, oWidth, oDepth);
            return result;
        }

        throw new Exception("Failed to calc Dot product");
    }

    public static float Max(Tensor one)
    {
        var (height, width, depth) = one.Shape;
        var bestVal = float.MinValue;
        var dim = one.GetDimension();
        
        if (dim == 1)
        {
            Tensor.TensorLoop(i =>
            {
                if (one[i] > bestVal)
                {
                    bestVal = one[i];
                }
            }, height);
        }

        if (dim == 2)
        {
            Tensor.TensorLoop((i, j) =>
            {
                if (one[i, j] > bestVal)
                {
                    bestVal = one[i, j];
                }
            }, height, width);
        }

        if (dim == 3)
        {
            Tensor.TensorLoop((i, j, k) =>
            {
                if (one[i, j, k] > bestVal)
                {
                    bestVal = one[i, j, k];
                }
            }, height, width, depth);
        }

        return bestVal > float.MinValue ? bestVal : 0;
    }

    public static float HistEntropy(Tensor one)
    {
        if (one.Sum() == 0f)
        {
            return 0;
        }

        var bins = one
            .GroupBy(val => val)
            .ToDictionary(key => key.Key, val => (float)val.Count());
        
        return bins.Keys
            .Select(val => bins[val] / one.Count())
            .Where(p => p > 0)
            .Sum(p => p * (float)Math.Log(p, 2)) * -1f;
    }
}