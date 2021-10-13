using System.Collections.Generic;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat.Layers
{
    public interface ILayer
    {
        List<float> Forward(List<float> data);

        void Reset();

        LayerType GetType();

        (int inSize, int outSize) Shape();

        ILayer CloneLayer();
    }
}