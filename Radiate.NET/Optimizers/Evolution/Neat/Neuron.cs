using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Radiate.NET.Domain.Activation;

namespace Radiate.NET.Optimizers.Evolution.Neat
{
    public class Neuron
    {
        public NeuronId Id { get; set; }
        public NeuronType NeuronType { get; set; }
        public float ActivatedValue { get; set; }
        public float DeactivatedValue { get; set; }
        public float CurrentState { get; set; }
        public float Error { get; set; }
        public float Bias { get; set; }
        private List<EdgeId> Outgoing { get; set; }
        private List<NeuronLink> Incoming { get; set; }
        private IActivationFunction Activation { get; set; }

        public Neuron() { }

        public Neuron(NeuronId id, NeuronType neuronType, Activation activation)
        {
            Id = id;
            Outgoing = new List<EdgeId>();
            Incoming = new List<NeuronLink>();
            Activation = ActivationFunctionFactory.Get(activation);
            NeuronType = neuronType;
            ActivatedValue = 0;
            DeactivatedValue = 0;
            CurrentState = 0;
            Error = 0;
            Bias = (float) new Random().NextDouble();
        }

        public void AddIncoming(Edge edge)
        {
            Incoming.Add(new NeuronLink
            {
                Id = edge.Id,
                Src = edge.Src,
                Weight = edge.Weight
            });
        }

        public void AddOutgoing(EdgeId edge)
        {
            Outgoing.Add(edge);
        }

        public void UpdateIncoming(Edge edge, float weight)
        {
            Incoming.Find(link => link.Id.Equals(edge.Id)).Weight = weight;

        }

        public void RemoveIncoming(Edge edge)
        {
            var link = Incoming.Single(link => link.Id.Equals(edge.Id));
            Incoming.Remove(link);
        }

        public void RemoveOutgoing(EdgeId edge)
        {
            if (Outgoing.Any(going => going.Equals(edge)))
            {
                var foundEdge = Outgoing.SingleOrDefault(going => going.Equals(edge));
                Outgoing.Remove(foundEdge);
            }
        }

        public List<NeuronLink> IncomingEdges() => Incoming;

        public List<EdgeId> OutgoingEdges() => Outgoing;

        public void Activate()
        {
            ActivatedValue = Activation.Activate(CurrentState);
            DeactivatedValue = Activation.Deactivate(CurrentState);
        }

        public void Reset()
        {
            Error = 0;
            ActivatedValue = 0;
            DeactivatedValue = 0;
            CurrentState = 0;
        }

        public Neuron Clone() => new()
        {
            Id = new NeuronId { Index = Id.Index },
            Outgoing = Outgoing
                .Select(outgoing => new EdgeId
                {
                    Index = outgoing.Index
                })
                .ToList(),
            Incoming = Incoming
                .Select(incoming => new NeuronLink
                {
                    Id = new EdgeId { Index = incoming.Id.Index },
                    Src = new NeuronId { Index = incoming.Src.Index },
                    Weight = incoming.Weight
                })
                .ToList(),
            Activation = Activation,
            NeuronType = NeuronType,
            ActivatedValue = 0,
            DeactivatedValue = 0,
            CurrentState = 0,
            Error = 0,
            Bias = Bias
        };

    }
}