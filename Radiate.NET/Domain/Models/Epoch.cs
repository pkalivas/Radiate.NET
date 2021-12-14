using System.Collections.Generic;

namespace Radiate.NET.Domain.Models
{
    public class Epoch
    {
        public List<float[]> Predictions { get; set; }
        public List<float> IterationLoss { get; set; }
        public float Loss { get; set; }
        public float ClassificationAccuracy { get; set; }
        public float RegressionAccuracy { get; set; }
    }
}