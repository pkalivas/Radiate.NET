using System.Numerics;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Gradients;

public class Adam : IGradient
{
    private readonly float _alpha;
    private readonly float _betaOne;
    private readonly float _betaTwo;
    private readonly float _epsilon;

    public Adam(float alpha, float betaOne, float betaTwo, float epsilon)
    {
        _alpha = alpha;
        _betaOne = betaOne;
        _betaTwo = betaTwo;
        _epsilon = epsilon;
    }
    
    public Tensor Calculate(Tensor gradient, int epoch)
    {
        var bOneDiscount = Pow(_betaOne, epoch + 1);
        var bTwoDiscount = Pow(_betaTwo, epoch + 1);
        var lRate = (_alpha * Root(1 - bTwoDiscount)) / (1 - bOneDiscount);

        var gradientShape = gradient.Shape;
        var ms = Tensor.Fill(gradientShape, 0f);
        var vs = Tensor.Fill(gradientShape, 0f);

        return gradient.GetDimension() switch
        {
            1 => Calc1D(gradient, ms, vs, lRate, bOneDiscount, bTwoDiscount),
            2 => Calc2D(gradient, ms, vs, lRate, bOneDiscount, bTwoDiscount),
            3 => Calc3D(gradient, ms, vs, lRate, bOneDiscount, bTwoDiscount),
            _ => throw new Exception($"Cannot calc adam of more than 3D")
        };
    }

    private Tensor Calc3D(Tensor gradient, Tensor ms, Tensor vs, float lRate, float bOneDiscount, float bTwoDiscount)
    {
        var result = Tensor.Like(gradient.Shape);
        for (var i = 0; i < gradient.Shape.Height; i++)
        {
            for (var j = 0; j < gradient.Shape.Width; j++)
            {
                for (var k = 0; k < gradient.Shape.Depth; k++)
                {
                    var bGrad = gradient[i, j, k];
                    
                    ms[i, j, k] = ms[i, j, k] * _betaOne + (1 - _betaOne) * bGrad;
                    vs[i, j, k] = vs[i, j, k] * _betaTwo + (1 - _betaTwo) * (bGrad * bGrad);
                    
                    var mCap = ms[i, j, k] / (1 - bOneDiscount);
                    var vCap = vs[i, j, k] / (1 - bTwoDiscount);
                    
                    result[i, j, k] += lRate * (mCap / (Root(vCap) + _epsilon));
                }
            }
        }

        return result;
    }
    
    private Tensor Calc2D(Tensor gradient, Tensor ms, Tensor vs, float lRate, float bOneDiscount, float bTwoDiscount)
    {
        var result = Tensor.Like(gradient.Shape);
        for (var i = 0; i < gradient.Shape.Height; i++)
        {
            for (var j = 0; j < gradient.Shape.Width; j++)
            {
                var bGrad = gradient[i, j];
                
                ms[i, j] = ms[i, j] * _betaOne + (1 - _betaOne) * bGrad;
                vs[i, j] = vs[i, j] * _betaTwo + (1 - _betaTwo) * (bGrad * bGrad);
                
                var mCap = ms[i, j] / (1 - bOneDiscount);
                var vCap = vs[i, j] / (1 - bTwoDiscount);
                
                result[i, j] += lRate * (mCap / (Root(vCap) + _epsilon));
            }
        }

        return result;
    }
    
    private Tensor Calc1D(Tensor gradient, Tensor ms, Tensor vs, float lRate, float bOneDiscount, float bTwoDiscount)
    {
        var result = Tensor.Like(gradient.Shape);
        for (var i = 0; i < gradient.Shape.Height; i++)
        {
            var bGrad = gradient[i];
            
            ms[i] = ms[i] * _betaOne + (1 - _betaOne) * bGrad;
            vs[i] = vs[i] * _betaTwo + (1 - _betaTwo) * (bGrad * bGrad);
            
            var mCap = ms[i] / (1 - bOneDiscount);
            var vCap = vs[i] / (1 - bTwoDiscount);
            
            result[i] += lRate * (mCap / (Root(vCap) + _epsilon));
        }

        return result;
    }
    
    private static float Root(float value) =>
        value > 0f 
            ? (float) Math.Sqrt(value) 
            : (float) Complex.Sqrt(value).Imaginary;

    private static float Pow(float val, float exp) => (float)Math.Pow(val, exp);
}
