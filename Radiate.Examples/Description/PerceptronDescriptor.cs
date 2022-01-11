using Radiate.Domain.Models;
using Radiate.Domain.Models.Wraps;
using Radiate.Domain.Records;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;

namespace Radiate.Examples.Description;

public class PerceptronDescriptor : IDescribe
{
    public string Describe<T>(T data)
    {
        var wrap = data switch
        {
            MultiLayerPerceptron perc => perc.Save(),
            _ => throw new Exception($"Wrong object given to PerceptronDescriptor")
        };

        var network = wrap.MultiLayerPerceptronWrap;

        var totalParameters = 0;
        var result = "";
        
        var top = string.Format("{0,-10} | {1,-12} | {2,-10} | {3,-10} | {4,-10}\n", 
            "Layer", 
            "Shape", 
            "Kernel", 
            "Activation",
            "Parameters");
        
        var sep = string.Join("-", Enumerable.Range(0, top.Length).Select(_ => ""));
        result += $"\t{top}\t{sep}\n";
        
        foreach (var layer in network.LayerWraps)
        {
            var (desc, parameters) = layer.LayerType switch
            {
                LayerType.Conv => GetConv(layer.Conv),
                LayerType.Dense => GetDense(layer.Dense),
                LayerType.MaxPool => GetPool(layer.MaxPool),
                LayerType.Flatten => GetFlatten(layer.Flatten),
                LayerType.Dropout => GetDropout(layer.Dropout),
                LayerType.LSTM => GetLstm(layer.Lstm),
                _ => ("", 0)
            };

            totalParameters += parameters;
            result += desc;
        }

        result += "\t" + sep + "\n";
        result += Format("", null, null, "", totalParameters);
        
        return result;
    }

    private static (string, int) GetConv(ConvWrap conv)
    {
        var parameters = conv.Filters.Sum(f => f.Count());
        parameters += conv.Bias.Count();

        var result = Format("Conv", conv.Shape, conv.Kernel, conv.Activation.ToString(), parameters);

        return (result, parameters);
    }

    private static (string, int) GetDense(DenseWrap dense)
    {
        var parameters = dense.Weights.Count() + dense.Bias.Count();
        var result = Format("Dense", dense.Shape, null, dense.Activation.ToString(), parameters);

        return (result, parameters);
    }

    private static (string, int) GetPool(MaxPoolWrap pool)
    {
        var result = Format("MaxPool", pool.Shape, pool.Kernel);

        return (result, 0);
    }

    private static (string, int) GetFlatten(FlattenWrap flat)
    {
        var result = Format("Flatten", flat.Shape);
        return (result, 0);
    }

    private static (string, int) GetDropout(DropoutWrap drop)
    {
        var result = Format("Dropout");
        return (result, 0);
    }

    private static (string, int) GetLstm(LSTMWrap wrap)
    {
        var parameters = 0;
        parameters += wrap.ForgetGate.Weights.Count() + wrap.ForgetGate.Bias.Count();
        parameters += wrap.GateGate.Weights.Count() + wrap.GateGate.Bias.Count();
        parameters += wrap.InputGate.Weights.Count() + wrap.InputGate.Bias.Count();
        parameters += wrap.OutputGate.Weights.Count() + wrap.OutputGate.Bias.Count();

        var result = Format("LSTM", wrap.Shape, null, wrap.HiddenActivation.ToString(), parameters);

        return (result, parameters);
    }

    private static string Format(string layerType, Shape shape = null, Kernel kernel = null, string act = "", int parameters = 0)
    {
        var (height, width, depth) = shape ?? new Shape(0);


        return string.Format("\t{0,-10} | {1,-12} | {2,-10} | {3,-10} | {4,-10:N0}\n",
            layerType,
            shape == null ? "" : $"{height}, {width}, {depth}",
            kernel == null ? "" : $"{kernel.Count}, {kernel.Dim}",
            act,
            parameters);
    }
}