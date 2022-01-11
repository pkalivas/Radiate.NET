﻿using Radiate.Data;
using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Gradients;
using Radiate.Domain.Tensors;
using Radiate.Domain.Tensors.Enums;
using Radiate.Examples.Callbacks;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.SVM;
using Radiate.Optimizers.Supervised.SVM.Info;

namespace Radiate.Examples.Examples;

public class SVMPredictor : IExample
{
    public async Task Run()
    {
        const int maxEpoch = 30;

        var (rawInputs, rawLabels) = await new BreastCancer().GetDataSet();

        var pair = new TensorTrainSet(rawInputs, rawLabels)
            .Kernel(FeatureKernel.Linear)
            .TransformFeatures(Norm.Standardize)
            .Split()
            .Shuffle();
        
        var info = new SVMInfo(pair.InputShape, pair.OutputCategories, 1e-3f);
        var svm = new SupportVectorMachine(info, new GradientInfo
        {
            Gradient = Gradient.SGD,
            LearningRate = 1e-3f
        });
        
        var optimizer = new Optimizer<SupportVectorMachine>(svm, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair, maxEpoch, false),
            new ModelWriterCallback(),
            new ConfusionMatrixCallback(),
        });

        await optimizer.Train(epoch => epoch.Index == maxEpoch);
    }
}