using Radiate.Callbacks.Interfaces;
using Radiate.Extensions;
using Radiate.Optimizers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Callbacks;

public class FreeStyleCallback : ITrainingCompletedCallback
{
    private readonly int _freeStyleAmount;

    public FreeStyleCallback(int freeStyleAmount = 10)
    {
        _freeStyleAmount = freeStyleAmount;
    }
    
    public async Task CompleteTraining(Optimizer optimizer, TensorTrainSet tensorSet)
    {
        var currentOutput = new Prediction(Array.Empty<float>().ToTensor(), 0, 0);
        foreach (var (feature, target) in tensorSet.InputsToTensorRow())
        {
            currentOutput = optimizer.Predict(feature.ToArray());
            Console.WriteLine($"{feature.Max()} {target.Max()} : {currentOutput.Confidence:N2}");
        }

        for (var i = 0; i < _freeStyleAmount; i++)
        {
            var input = (int)Math.Round(currentOutput.Confidence);
            currentOutput = optimizer.Predict(new[] { (float)input });
            Console.WriteLine($"{input} : {currentOutput.Confidence:N2}");
        }
    }
}