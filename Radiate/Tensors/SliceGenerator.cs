using Radiate.Records;

namespace Radiate.Tensors;

public class SliceGenerator
{
    private readonly Kernel _kernel;
    private readonly Shape _kernelShape;
    private readonly int _stride;

    public SliceGenerator(Kernel kernel, int depth, int stride)
    {
        _kernel = kernel;
        _kernelShape = new Shape(kernel.Dim, kernel.Dim, depth);
        _stride = stride;
    }

    public int Stride => _stride;

    public IEnumerable<Slice> Slice(Tensor volume)
    {
        var (_, _, kDepth) = volume.Shape;
        var (hStride, wStride) = CalcStride(volume);
        
        for (var i = 0; i < hStride; i++)
        {
            for (var j = 0; j < wStride; j++)
            {
                var sliceH = new[] { i, i + _kernel.Dim };
                var sliceW = new[] { j, j + _kernel.Dim };
                var sliceD = new[] { 0, kDepth };
                var tensorSlice = volume.Slice(sliceH, sliceW, sliceD);

                yield return new Slice(tensorSlice, i, j);
            }
        }
    }

    public IEnumerable<Slice> Slice3D(Tensor volume)
    {
        var (_, _, vDepth) = volume.Shape;
        var (hStride, wStride) = CalcStride(volume);

        for (var i = 0; i < hStride; i++)
        {
            for (var j = 0; j < wStride; j++)
            {
                for (var k = 0; k < vDepth; k++)
                {
                    var hStrideBase = i * _stride;
                    var hStrideLimit = hStrideBase + _kernel.Dim;

                    var wStrideBase = j * _stride;
                    var wStrideLimit = wStrideBase + _kernel.Dim;
                    
                    var heightChange = new[] { hStrideBase, hStrideLimit };
                    var widthChange = new[] { wStrideBase, wStrideLimit };
                    var depthChange = new[] { k, k + 1 };

                    var tensorSlice = volume.Slice(heightChange, widthChange, depthChange);

                    yield return new Slice(tensorSlice, i, j, k);
                }
            }
        }
    }

    public (int hStride, int wStride) CalcStride(Tensor input)
    {
        var (vHeight, vWidth, _) = input.Shape;
        var (kHeight, kWidth, _) = _kernelShape;
        
        var hStride = ((vHeight - kHeight) / _stride) +1;
        var wStride = ((vWidth - kWidth) / _stride) + 1;
        
        return (hStride, wStride);
    }


}