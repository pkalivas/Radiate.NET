using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;

namespace Radiate.Examples.Callbacks;

public class ConfusionMatrixCallback : ITrainingCompletedCallback
{
    public Task CompleteTraining<T>(Optimizer<T> optimizer, List<Epoch> epochs, TensorTrainSet trainSet) where T : class
    {
        var result = new Tensor(trainSet.OutputCategories, trainSet.OutputCategories);
        var (trainIns, trainOuts) = trainSet.RawTrainingInputs();
        foreach (var (x, y) in trainIns.Zip(trainOuts))
        {
            var pred = optimizer.Predict(x);
            var idx = pred.Classification;
            result[(int)y.Max(), idx]++;
        }

        Console.WriteLine("\n\nTraining:");
        Console.WriteLine($"{result.ConfusionMatrix()}");
        
        result = new Tensor(trainSet.OutputCategories, trainSet.OutputCategories);
        var (testIns, testOuts) = trainSet.RawTestingInputs();
        foreach (var (x, y) in testIns.Zip(testOuts))
        {
            var pred = optimizer.Predict(x);
            var idx = pred.Classification;
            result[(int)y.Max(), idx]++;
        }

        Console.WriteLine("Testing");
        Console.WriteLine($"{result.ConfusionMatrix()}");
        
        return Task.CompletedTask;
    }

   
}