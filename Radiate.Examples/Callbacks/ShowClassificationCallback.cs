using Radiate.Callbacks.Interfaces;
using Radiate.Optimizers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Examples.Callbacks;

public class ShowClassificationCallback : ITrainingCompletedCallback
{
    public Task CompleteTraining<T>(Optimizer<T> optimizer, List<Epoch> epochs, TensorTrainSet tensorSet) where T : class
    {
        Console.WriteLine("\nTraining");
        var (trainIns, trainOuts) = tensorSet.RawTrainingInputs();
        foreach (var (x, y) in trainIns.Zip(trainOuts))
        {
            var pred = optimizer.Predict(x);
            Console.WriteLine($"[{pred.Classification}, {y.Max()}]");
        }
        
        Console.WriteLine($"\nTesting");
        var (testIns, testOuts) = tensorSet.RawTestingInputs();
        foreach (var (x, y) in testIns.Zip(testOuts))
        {
            var pred = optimizer.Predict(x);
            Console.WriteLine($"[{pred.Classification}, {y.Max()}]");
        }
        
        return Task.CompletedTask;
    }
}