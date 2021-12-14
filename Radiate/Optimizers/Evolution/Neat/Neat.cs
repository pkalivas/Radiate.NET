﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Domain.Activation;
using Radiate.Optimizers.Evolution.Engine;

namespace Radiate.Optimizers.Evolution.Neat
{
    public class Neat : Genome
    {
        private NeuronId[] Inputs { get; set; }
        private NeuronId[] Outputs { get; set; }
        private List<Neuron> Nodes { get; set; }
        private List<Edge> Edges { get; set; }
        private Dictionary<Guid, EdgeId> EdgeInnovationLookup { get; set; }
        private Tracer Tracer { get; set; }
        private Activation Activation { get; set; }
        private bool FastMode { get; set; }

        private Neat() { }
        

        public Neat(int inputSize, int outputSize, Activation activation)
        {
            Inputs = new NeuronId[inputSize];
            Outputs = new NeuronId[outputSize];
            Nodes = new List<Neuron>();
            Edges = new List<Edge>();
            EdgeInnovationLookup = new Dictionary<Guid, EdgeId>();
            Activation = activation;
            FastMode = true;

            for (var i = 0; i < inputSize; i++)
            {
                Inputs[i] = MakeNode(NeuronType.Input, activation);
            }

            for (var i = 0; i < outputSize; i++)
            {
                Outputs[i] = MakeNode(NeuronType.Output, activation);
            }

            var random = new Random();
            foreach (var input in Inputs)
            {
                foreach (var output in Outputs)
                {
                    var weight = random.NextDouble() * 2 - 1;
                    MakeEdge(input, output, (float) weight);
                }
            }
        }
        
        
        public List<float> GetOutputs() => 
            Outputs
                .Select(output => Nodes[output.Index].ActivatedValue)
                .ToList();

        public bool HasTracer() => Tracer is not null;
        
        private void AddNode(Activation activation)
        {
            FastMode = false;

            var newNodeId = MakeNode(NeuronType.Hidden, activation);
            var currentEdge = Edges[new Random().Next(Edges.Count)];

            MakeEdge(currentEdge.Src, newNodeId, 1.0f);
            MakeEdge(newNodeId, currentEdge.Dst, currentEdge.Weight);
            
            Edges[currentEdge.Id.Index].Disable(Nodes);
        }
        
        private void AddEdge()
        {
            var sending = RandomNodeNotOfType(NeuronType.Output);
            var receiving = RandomNodeNotOfType(NeuronType.Input);

            if (ValidConnection(sending, receiving))
            {
                MakeEdge(sending, receiving, (float) new Random().NextDouble());
                FastMode = false;
            }
        }

        private NeuronId MakeNode(NeuronType neuronType, Activation activation)
        {
            var nodeId = new NeuronId { Index = Nodes.Count };
            Nodes.Add(new Neuron(nodeId, neuronType, activation));
            return nodeId;
        }

        private EdgeId MakeEdge(NeuronId src, NeuronId dst, float weight)
        {
            var edgeId = new EdgeId { Index = Edges.Count };
            var edge = new Edge(edgeId, src, dst, weight, true);
            
            edge.LinkNodes(Nodes);
            EdgeInnovationLookup[edge.Innovation] = edgeId;
            Edges.Add(edge);

            return edgeId;
        }
        
        private bool ValidConnection(NeuronId sending, NeuronId receiving)
        {
            if (sending.Equals(receiving))
            {
                return false;
            }
            
            if (Exists(sending, receiving))
            {
                return false;
            } 
            
            if (Cyclical(sending, receiving))
            {
                return false;
            }

            return true;
        }

        private bool Cyclical(NeuronId sending, NeuronId receiving)
        {
            var receivingNode = Nodes[receiving.Index];
            var stack = receivingNode.OutgoingEdges()
                .Select(edge => Edges[edge.Index].Dst.Index)
                .ToList();

            while (stack.Any())
            {
                var topNode = stack.Last();
                
                stack.RemoveAt(stack.Count - 1);

                var current = Nodes[topNode];
                if (current.Id.Equals(sending))
                {
                    return true;
                }

                stack.AddRange(current.OutgoingEdges().Select(edge => Edges[edge.Index].Dst.Index));
            }

            return false;
        }

        private bool Exists(NeuronId sending, NeuronId receiving) => 
            Edges.Any(edge => edge.Src.Equals(sending) && edge.Dst.Equals(receiving));


        private NeuronId RandomNodeNotOfType(NeuronType neuronType)
        {
            var random = new Random();
            var node = Nodes[random.Next(Nodes.Count)];

            while (node.NeuronType == neuronType)
            {
                node = Nodes[random.Next(Nodes.Count)];
            }

            return node.Id;
        }


        private void EditWeights(float editable, float size)
        {
            var random = new Random();
            foreach (var edge in Edges)
            {
                var shouldEdit = random.NextDouble() < editable;
                var weightValue = shouldEdit 
                    ? (float) random.NextDouble() 
                    : edge.Weight * ((float) random.NextDouble() * size - size);
                edge.UpdateWeight(weightValue, Nodes);
            }

            foreach (var node in Nodes)
            {
                var shouldEdit = random.NextDouble() < editable;
                node.Bias = shouldEdit ? (float) random.NextDouble() : node.Bias * ((float) random.NextDouble() * size - size);
            }
        }

        private void UpdateTracer()
        {
            if (Tracer is not null)
            {
                foreach (var node in Nodes)
                {
                    Tracer.UpdateNeuronActivation(node);
                    Tracer.UpdateNeuronDerivative(node);
                }

                Tracer.Index++;
            }
        }


        private float[] FastForward(IReadOnlyList<float> data)
        {
            var inSize = Inputs.Length;

            for (var i = 0; i < inSize; i++)
            {
                Nodes[i].Reset();
                Nodes[i].ActivatedValue = data[i];
            }

            var result = new List<float>();
            for (var i = inSize; i < Nodes.Count; i++)
            {
                Nodes[i].Reset();
                
                Nodes[i].CurrentState = Nodes[i].IncomingEdges()
                    .Select((neuron, idx) => (neuron, data[idx]))
                    .Aggregate(Nodes[i].Bias, (sum, current) => sum + (current.Item2 * current.neuron.Weight));
                
                Nodes[i].Activate();

                result.Add(Nodes[i].ActivatedValue);
            }
            
            UpdateTracer();

            return result.ToArray();
        }
        
        
        public override float[] Forward(float[] data)
        {
            if (Inputs.Length != data.Length)
            {
                throw new Exception($"Dense layer input does not match. Input size {Inputs.Length}, given size {data.Length}");
            }

            if (FastMode)
            {
                return FastForward(data);
            }

            var outputs = new List<float>();
            var updates = new List<NodeUpdate>();
            var pendingCount = 0;
            var lowestPendingIdx = Nodes.Count;

            var inputCounter = 0;
            foreach (var node in Nodes)
            {
                node.Reset();

                var update = new NodeUpdate();
                if (node.NeuronType == NeuronType.Input)
                {
                    var value = data[inputCounter++];
                    node.ActivatedValue = value;

                    update = new NodeUpdate
                    {
                        UpdateType = UpdateType.Activated,
                        Activated = new Activated
                        {
                            Value = value,
                        }
                    };
                }

                if (node.NeuronType == NeuronType.Output)
                {
                    var processed = NodeUpdate.Process(updates, node, outputs.Count);
                    var activatedValue = processed.IsActivated();

                    outputs.Add(activatedValue.value);

                    update = processed;
                }

                if (node.NeuronType == NeuronType.Hidden)
                {
                    update = NodeUpdate.Process(updates, node, 0);
                }

                if (update.IsPending())
                {
                    var idx = updates.Count;
                    if (idx < lowestPendingIdx)
                    {
                        lowestPendingIdx = idx;
                    }

                    pendingCount++;
                }

                updates.Add(update);
            }


            // Step two
            var maxTries = 10;
            while (pendingCount > 0)
            {
                var changes = 0;

                var startIdx = lowestPendingIdx;
                var endIdx = Nodes.Count;
                lowestPendingIdx = endIdx;

                for (var i = startIdx; i < endIdx; i++)
                {
                    var node = Nodes[i];
                    var oldUpdate = updates[i];

                    if (oldUpdate.IsPending())
                    {
                        var outputIdx = oldUpdate.Output();
                        var update = NodeUpdate.Process(updates, node, outputIdx);

                        if (update.UpdateType == UpdateType.Pending && i < lowestPendingIdx)
                        {
                            lowestPendingIdx = i;
                        }
                        else
                        {
                            if (update.UpdateType == UpdateType.Activated)
                            {
                                var idx = update.Activated.Output;
                                outputs[idx] = update.Activated.Value;
                            }

                            pendingCount--;
                            changes++;
                        }

                        updates[i] = update;
                    }
                }

                if (changes == 0)
                {
                    maxTries--;
                    if (maxTries == 0)
                    {
                        throw new Exception($"Failed forward pass in NEAT.");
                    }
                }
            }
            
            UpdateTracer();

            return outputs.ToArray();
        }


        public override async Task<Genome> Crossover(Genome other, EvolutionEnvironment environment, double crossoverRate)
        {
            var random = new Random();

            var child = CloneGenome() as Neat;
            var parentTwo = other as Neat;
            var neatEnv = environment as NeatEnvironment;

            if (random.NextDouble() < crossoverRate)
            {
                foreach (var edge in child.Edges)
                {
                    if (parentTwo.EdgeInnovationLookup.ContainsKey(edge.Innovation))
                    {
                        var parentEdge = parentTwo.Edges.Single(pedge => pedge.Innovation == edge.Innovation);

                        if (random.NextDouble() < .5)
                        {
                            edge.UpdateWeight(parentEdge.Weight, child.Nodes);
                        }

                        if ((!edge.Active || !parentEdge.Active) && random.NextDouble() < neatEnv.ReactivateRate)
                        {
                            edge.Enable(child.Nodes);
                        }
                    }
                }
            }
            else
            {
                if (random.NextDouble() < neatEnv.WeightMutateRate)
                {
                    child.EditWeights(neatEnv.EditWeights, neatEnv.WeightPerturb);
                }

                if (random.NextDouble() < neatEnv.NewNodeRate)
                {
                    var actFunction = neatEnv.ActivationFunctions[random.Next(neatEnv.ActivationFunctions.Count)];
                    child.AddNode(actFunction);
                }

                if (random.NextDouble() < neatEnv.NewEdgeRate)
                {
                    child.AddEdge();
                }
            }

            return child;
        }

        public override async Task<double> Distance(Genome other, EvolutionEnvironment environment)
        {
            var parentTwo = other as Neat;

            var similar = EdgeInnovationLookup.Keys.Count(innov => parentTwo.EdgeInnovationLookup.ContainsKey(innov));

            var oneScore = similar / Edges.Count;
            var twoScore = similar / parentTwo.Edges.Count;

            return 2 - (oneScore + twoScore);
        }

        public override Genome CloneGenome() => new Neat
        {
            Inputs = Inputs
                .Select(input => new NeuronId { Index = input.Index })
                .ToArray(),
            Outputs = Outputs
                .Select(output => new NeuronId { Index = output.Index })
                .ToArray(),
            Nodes = Nodes
                .Select(node => node.Clone())
                .ToList(),
            Activation = Activation,
            Edges = Edges
                .Select(edge => new Edge
                {
                    Active = edge.Active,
                    Dst = new NeuronId { Index = edge.Dst.Index },
                    Id = new EdgeId { Index = edge.Id.Index },
                    Innovation = edge.Innovation,
                    Src = new NeuronId { Index = edge.Src.Index },
                    Weight = edge.Weight
                })
                .ToList(),
            EdgeInnovationLookup = EdgeInnovationLookup
                .Select(pair => (Id: pair.Key, edge: new EdgeId { Index = pair.Value.Index }))
                .ToDictionary(key => key.Id, val => val.edge),
            FastMode = Nodes.Count == Inputs.Length + Outputs.Length,
        };

        public override void ResetGenome()
        {
            foreach (var node in Nodes)
            {
                node.Reset();
            }

            if (Tracer is not null)
            {
                Tracer = new();
            }
        }

    }
}