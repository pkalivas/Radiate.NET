
namespace Radiate.Gradients;

public class GradientInfo
{
    public GradientInfo()
    {
        Gradient = Gradient.Adam;
        LearningRate = 0.01f;
        BetaOne = 0.9f;
        BetaTwo = 0.999f;
        Epsilon = 1e-8f;
    }
    
    public Gradient Gradient { get; set; }
    public float LearningRate { get; set; }
    public float BetaOne { get; set; }
    public float BetaTwo { get; set; }
    public float Epsilon { get; set; }
}