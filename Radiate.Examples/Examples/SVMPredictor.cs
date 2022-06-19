using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Gradients;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.SVM;
using Radiate.Optimizers.Supervised.SVM.Info;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

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
        
        var optimizer = new Optimizer(svm, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair, maxEpoch, false),
            new ModelWriterCallback(),
            new ConfusionMatrixCallback(),
        });

        await optimizer.Train<SupportVectorMachine>(epoch => epoch.Index == maxEpoch);
    }
}