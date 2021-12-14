using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Services;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Perceptrons;

namespace Radiate.Optimizers
{
    public class Optimizer
    {
        private const int DefaultBatchSize = 1;
        
        private readonly IOptimizer _optimizer;
        private readonly ILossFunction _lossFunction;
        private readonly GradientInfo _gradientInfo;
        private readonly Shape _shape;

        public Optimizer(
            IOptimizer optimizer, 
            Loss lossFunction) : this(optimizer, lossFunction, new(0, 0, 0), new GradientInfo()) { }
        
        public Optimizer(
            IOptimizer optimizer, 
            Loss lossFunction, 
            Shape shape) : this(optimizer, lossFunction, shape, new GradientInfo()) { }
        
        public Optimizer(
            IOptimizer optimizer, 
            Loss lossFunction, 
            GradientInfo gradient) : this(optimizer, lossFunction, new(0, 0, 0), gradient) { }

        public Optimizer(IOptimizer optimizer, Loss lossFunction, Shape shape, GradientInfo gradientInfo)
        {
            _optimizer = optimizer;
            _gradientInfo = gradientInfo;
            _shape = shape;
            _lossFunction = LossFunctionFactory.Get(lossFunction);
        }


        public async Task<List<Epoch>> Train(List<float[]> features, List<float[]> targets, Func<List<Epoch>, bool> trainFunc) =>
            await Train(features, targets, DefaultBatchSize, trainFunc);
        
        public async Task<List<Epoch>> Train(List<float[]> features, List<float[]> targets, int batchSize, Func<List<Epoch>, bool> trainFunc)
        {
            var batches = BatchService.CreateBatches(features, targets, _shape, batchSize);

            var epochs = new List<Epoch>();

            while (true)
            {
                var predictions = new List<float[]>();
                var epochErrors = new List<Cost>();

                foreach (var (inputs, answers) in batches)
                {
                    var batchErrors = new List<Cost>();
                    foreach (var (x, y) in inputs.Zip(answers))
                    {
                        var prediction = _optimizer.PassForward(x).Read1D();
                        
                        batchErrors.Add(_lossFunction.Calculate(prediction, y.Read1D()));
                        predictions.Add(prediction);
                    }

                    foreach (var (passError, _) in batchErrors.Select(pair => pair).Reverse())
                    {
                        _optimizer.PassBackward(new Tensor(passError), epochs.Count);
                    }

                    await _optimizer.Update(_gradientInfo, epochs.Count);
                    
                    epochErrors.AddRange(batchErrors);
                }
                
                epochs.Add(new Epoch
                {
                    Predictions = predictions,
                    Loss = epochErrors.Sum(err => err.loss) / epochErrors.Count,
                    IterationLoss = epochErrors.Select(err => err.loss).ToList(),
                    ClassificationAccuracy = ValidationService.ClassificationAccuracy(predictions, targets),
                    RegressionAccuracy = ValidationService.RegressionAccuracy(predictions, targets)
                });
                
                if (trainFunc(epochs))
                {
                    break;
                }
            }

            return epochs;
        }

        public Epoch Validate(List<float[]> features, List<float[]> targets)
        {
            var iterationLoss = new List<float>();
            var predictions = new List<float[]>();
            var batches = BatchService.CreateBatches(features, targets, _shape, DefaultBatchSize);
            foreach (var (input, answer) in batches)
            {
                foreach (var (feature, target) in input.Zip(answer))
                {
                    var prediction = _optimizer.Predict(feature);
                    var (_, loss) = _lossFunction.Calculate(prediction.Read1D(), target.Read1D());
                
                    iterationLoss.Add(loss);
                    predictions.Add(prediction.Read1D());   
                }
            }

            return new Epoch
            {
                Predictions = predictions,
                Loss = iterationLoss.Sum(),
                IterationLoss = iterationLoss,
                ClassificationAccuracy = ValidationService.ClassificationAccuracy(predictions, targets),
                RegressionAccuracy = ValidationService.RegressionAccuracy(predictions, targets)
            };
        }

        public Prediction Predict(float[] input)
        {
            var predIn = BatchService.Transform(input, _shape);
            var passResult = _optimizer.Predict(predIn).Read1D();
            var confidence = passResult.Max();
            var classification = passResult.ToList().IndexOf(confidence);

            return new Prediction(passResult, classification, confidence);
        }

    }
}