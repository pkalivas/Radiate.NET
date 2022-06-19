using Radiate.Callbacks.Interfaces;
using Radiate.Optimizers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Callbacks;

public class ShowRegressionCallback : ITrainingCompletedCallback
{
    public Task CompleteTraining(Optimizer optimizer, TensorTrainSet tensorSet)
    {
        Console.WriteLine("\nTraining");
        var (trainIns, trainOuts) = tensorSet.RawTrainingInputs();
        foreach (var (x, y) in trainIns.Zip(trainOuts))
        {
            var pred = optimizer.Predict(x);
            Console.WriteLine($"[{pred.Confidence:0.0000}, {y.Max()}]");
        }
        
        Console.WriteLine("\nTesting");
        var (testIns, testOuts) = tensorSet.RawTestingInputs();
        foreach (var (x, y) in testIns.Zip(testOuts))
        {
            var pred = optimizer.Predict(x);
            Console.WriteLine($"[{pred.Confidence:0.0000}, {y.Max()}]");
        }
        
        return Task.CompletedTask;
    }
}