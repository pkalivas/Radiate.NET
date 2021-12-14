using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Domain.Gradients;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers
{
    public class SoftMax : Layer
    {
        private readonly Stack<Tensor> _inputs;
        private readonly Stack<Tensor> _outputs;
        private readonly Tensor _weights;
        private readonly Tensor _bias;
        private readonly Tensor _weightGradients;
        private readonly Tensor _biasGradients;
        
        public SoftMax(Shape shape) : base(shape)
        {
            _weights = Tensor.Random2D(shape.Width, shape.Height);
            _bias = Tensor.Fill(new Shape(shape.Width, 0, 0), 0); 
            _inputs = new Stack<Tensor>();
            _outputs = new Stack<Tensor>();
            _weightGradients = new float[shape.Width, shape.Height].ToTensor();
            _biasGradients = new float[shape.Width].ToTensor();

            for (var i = 0; i < shape.Width; i++)
            {
                for (var j = 0; j < shape.Height; j++)
                {
                    _weights[i, j] /= shape.Height;
                }
            }
        }

        public override Tensor Predict(Tensor pass)
        {
            var input = pass.Flatten();
            var expTotal = 0f;
            var dotProduct = new float[Shape.Width].ToTensor();
            for (var i = 0; i < Shape.Width; i++)
            {
                dotProduct[i] = _bias[i] + input.Read1D()
                    .Select((inVal, idx) => _weights[i, idx] * inVal)
                    .Sum();

                expTotal += (float)Math.Exp(dotProduct[i]);
            }
            
            var result = new float[Shape.Width].ToTensor();
            for (var i = 0; i < dotProduct.Read1D().Length; i++)
            {
                result[i] /= expTotal;
            }

            return result;
        }

        public override Tensor FeedForward(Tensor pass)
        {
            var input = pass.Flatten();
            var expTotal = 0f;
            var dotProduct = new float[Shape.Width].ToTensor();
            for (var i = 0; i < Shape.Width; i++)
            {
                dotProduct[i] = _bias[i] + input.Read1D()
                    .Select((inVal, idx) => _weights[i, idx] * inVal)
                    .Sum();

                expTotal += (float)Math.Exp(dotProduct[i]);
            }
            
            var result = new float[Shape.Width].ToTensor();
            for (var i = 0; i < dotProduct.Read1D().Length; i++)
            {
                result[i] = (float)Math.Exp(dotProduct[i]) / expTotal;
            }
            
            _inputs.Push(pass);
            _outputs.Push(dotProduct);
            
            var expSum = dotProduct.Read1D().Sum(val => (float)Math.Pow(Math.E, val));
            
            return dotProduct.Read1D()
                .Select(val => (float)Math.Pow(Math.E, val) / expSum)
                .ToArray()
                .ToTensor();
            
            return result;
        }

        public override Tensor PassBackward(Tensor pass)
        {
            var oneD = pass.Read1D();
            var error = pass.Read1D().Single(p => p != 0);
            var previousOut = _outputs.Pop();
            var previousIn = _inputs.Pop();
            

            var prevExp = previousOut.Read1D().Select(val => (float)Math.Exp(val)).ToArray();
            var prevSum = prevExp.Sum();
            
            var deltaGrad = new float[Shape.Width];
            var previousFlat = previousIn.Flatten();
            var resultError = Tensor.Like(previousFlat.Shape);
            
            for (var i = 0; i < Shape.Width; i++)
            {
                if (pass[i] == error)
                {
                    deltaGrad[i] = prevExp[i] * (prevSum - prevExp[i]) / (prevSum * prevSum);
                }
                else
                {
                    deltaGrad[i] = -prevExp[i] * prevSum / (prevSum * prevSum);
                }
            
                _biasGradients[i] += deltaGrad[i] * error;
            
                for (var j = 0; j < Shape.Height; j++)
                {
                    _weightGradients[i, j] += deltaGrad[i] * error * previousFlat[j];
                    resultError[j] += _weights[i, j] * deltaGrad[i];
                }
            }
            
            return resultError.Reshape(previousIn.Shape);
        }

        public override Task UpdateWeights(GradientInfo gradient, int epoch)
        {
            var grad = GradientFactory.Get(gradient);

            _bias.Add(grad.Calculate(_biasGradients, epoch));
            _weights.Add(grad.Calculate(_weightGradients, epoch));
            _weightGradients.Zero();
            _biasGradients.Zero();

            return Task.CompletedTask;
        }

        private static Tensor CalcD(Tensor errors)
        {
            var values = errors.Read1D();
            var diagMatrix = values
                .Select((val, idx) => Enumerable
                    .Range(0, values.Length)
                    .Select(num => num == idx ? val : 0)
                    .ToArray())
                .ToArray();

            var tiledMatrix = values
                .Select(val => Enumerable
                    .Range(0, values.Length)
                    .Select(_ => val)
                    .ToArray())
                .ToArray();

            var transposedMatrix = values
                .Select(_ => values.Select(val => val).ToArray())
                .ToArray();
            
            var result = new float[values.Length];
            for (var i = 0; i < values.Length; i++)
            {
                for (var j = 0; j < values.Length; j++)
                {
                    result[i] += diagMatrix[j][i] - (tiledMatrix[j][i] * transposedMatrix[j][i]);
                }
            }

            return result.ToTensor();

        }
        
    }
}


            
// for (var i = 0; i < pass.Shape.Height; i++)
// {
//     if (pass[i] == 0)
//     {
//         continue;
//     }
//
//     var flattened = previousIn.Flatten().ElementsOneD.Sum();
//     var result = Tensor.Fill(_weightGradients.Shape, 0f);
//
//     var totalsExp = previousOut.ElementsOneD.Select(val => (float)Math.Exp(val)).ToArray();
//     var sumExp = totalsExp.Sum();
//                 
//
//     var t = new List<float>();
//     for (var j = 0; j < _weightGradients.Shape.Height; j++)
//     {
//         for (var k = 0; k < _weightGradients.Shape.Width; k++)
//         {
//             if (j == i)
//             {
//                 var grad = -totalsExp[j] * sumExp / (sumExp * sumExp);
//                 _weightGradients[j, k] = grad * pass[i];
//             }
//             else
//             {
//                 var grad = totalsExp[i] * (sumExp - totalsExp[i]) / (sumExp * sumExp);
//                 _weightGradients[j, k] = grad * pass[i] * flattened;
//             }
//                         
//                         
//             result[j, k] += _weights[j, k] * _weightGradients[j, k];
//         }
//     }
//
//
//     return Tensor.Reshape(result.Flatten(), previousIn.Shape);
// }
//
