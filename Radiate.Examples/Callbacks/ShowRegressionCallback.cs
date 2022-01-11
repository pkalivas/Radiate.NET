﻿using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;

namespace Radiate.Examples.Callbacks;

public class ShowRegressionCallback : ITrainingCompletedCallback
{
    public Task CompleteTraining<T>(Optimizer<T> optimizer, List<Epoch> epochs, TensorTrainSet tensorSet) where T : class
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