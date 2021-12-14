namespace Radiate.Domain.Records
{
    public record Prediction(float[] Result, int Classification, float Confidence);
}