using System.Collections.Generic;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat.Layers
{
    public interface ILayer
    {
        List<float> Forward(List<float> data);
        List<float> Backward(List<float> errors, float learningRate);
        void AddTracer();
        void RemoveTracer();
        void Reset();
        LayerType GetLayerType();
        ILayer CloneLayer();
    }
}