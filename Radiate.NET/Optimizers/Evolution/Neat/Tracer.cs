using System.Collections.Generic;

namespace Radiate.NET.Optimizers.Evolution.Neat
{
    public class Tracer
    {
        private Dictionary<NeuronId, List<float>> NeuronActivation { get; set; }
        private Dictionary<NeuronId, List<float>> NeuronDerivative { get; set; }
        public int Index { get; set; }

        public Tracer()
        {
            NeuronActivation = new();
            NeuronDerivative = new();
            Index = 0;
        }

        public void UpdateNeuronActivation(Neuron neuron)
        {
            if (NeuronActivation.ContainsKey(neuron.Id))
            {
                NeuronActivation[neuron.Id].Add(neuron.ActivatedValue);
            }
            else
            {
                NeuronActivation[neuron.Id] = new() { neuron.ActivatedValue };
            }
        }

        public void UpdateNeuronDerivative(Neuron neuron)
        {
            if (NeuronDerivative.ContainsKey(neuron.Id))
            {
                NeuronDerivative[neuron.Id].Add(neuron.DeactivatedValue);
            }
            else
            {
                NeuronDerivative[neuron.Id] = new() { neuron.DeactivatedValue };
            }
        }

        public float GetNeuronActivation(NeuronId neuron) => NeuronActivation[neuron][Index - 1];

        public float GetNeuronDerivative(NeuronId neuron) => NeuronDerivative[neuron][Index - 1];
    }
}