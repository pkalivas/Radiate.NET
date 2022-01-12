namespace Radiate.Tensors;

public static class Stacker
{
    public static Tensor StackZero(Tensor one, Tensor two)
    {
        var (oHeight, oWidth, rDepth) = one.Shape;
        var (tHeight, tWidth, tDepth) = two.Shape;
        var oneD = one.GetDimension();
        var twoD = two.GetDimension();
        
        if (oneD == 1 && twoD == 1)
        {
            var result = new Tensor(2, oHeight);
            Tensor.TensorLoop((i) => result[0, i] = one[i], oHeight);
            Tensor.TensorLoop((i) => result[1, i] = two[i], tHeight);
            return result;
        }
        
        if (oneD == 2 && twoD == 2)
        {
            var result = new Tensor(oHeight + tHeight, oWidth);
            Tensor.TensorLoop((i, j) => result[i, j] = one[i, j], oHeight, oWidth);
            Tensor.TensorLoop((i, j) => result[i + oHeight, j] = two[i, j], tHeight, tWidth);
            return result;
        }

        if (oneD == 1 && twoD == 2)
        {
            var result = new Tensor(tHeight + 1, oWidth);
            Tensor.TensorLoop((i) => result[0, i] = one[i], oHeight);
            Tensor.TensorLoop((i, j) => result[i + 1, j] = two[i, j], tHeight, tWidth);
            return result;
        }
        
        if (oneD == 2 && twoD == 1)
        {
            var result = new Tensor(oHeight + 1, oWidth);
            Tensor.TensorLoop((i, j) => result[i, j] = one[i, j], oHeight, oWidth);
            Tensor.TensorLoop((i) => result[oHeight, i] = two[i], tHeight);
            return result;
        }

        throw new Exception("Cannot stack tensors on Zero Axis - 3D is not implemented.");
    }

    public static Tensor StackOne(Tensor one, Tensor two)
    {
        var (oHeight, oWidth, rDepth) = one.Shape;
        var (tHeight, tWidth, tDepth) = two.Shape;
        var oneD = one.GetDimension();
        var twoD = two.GetDimension();
        
        if (oneD == 1 && twoD == 1)
        {
            var result = new Tensor(oHeight + tHeight);
            Tensor.TensorLoop((i) => result[i] = one[i], oHeight);
            Tensor.TensorLoop((i) => result[i + oHeight] = two[i], tHeight);
            return result;
        }
        
        if (oneD == 2 && twoD == 2)
        {
            var result = new Tensor(oHeight, oWidth + tWidth);
            Tensor.TensorLoop((i, j) => result[i, j] = one[i, j], oHeight, oWidth);
            Tensor.TensorLoop((i, j) => result[i, j + oWidth] = two[i, j], tHeight, tWidth);
            return result;
        }

        throw new Exception("Cannot stack tensors on One Axis - 3D is not implemented.");
    }
}