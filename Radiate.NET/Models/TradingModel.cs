using System.Collections.Generic;
using Radiate.NET.Enums;
using Radiate.NET.Models.Neat.Wraps;

namespace Radiate.NET.Models
{
    public interface ITradingModel
    {
        List<float> Forward(List<float> data);
    }


    public class TradingModel
    {
        public ModelType ModelType { get; set; }
        public NeatWrap NeatWrap { get; set; }
    }
}