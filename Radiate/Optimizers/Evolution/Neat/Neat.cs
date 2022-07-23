using Radiate.Activations;
using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Neat;

public class Neat : Allele, IGenome, IPredictionModel
{
    private readonly NeuronId[] _inputs;
    private readonly NeuronId[] _outputs;
    private readonly List<Neuron> _nodes;
    private readonly List<Edge> _edges;
    private readonly Dictionary<Guid, EdgeId> _edgeInnovationLookup;
    private readonly Activation _activation;
    
    public Neat(int inputSize, int outputSize, Activation activation)
    {
        _inputs = new NeuronId[inputSize];
        _outputs = new NeuronId[outputSize];
        _nodes = new List<Neuron>();
        _edges = new List<Edge>();
        _edgeInnovationLookup = new Dictionary<Guid, EdgeId>();
        _activation = activation;

        for (var i = 0; i < inputSize; i++)
        {
            _inputs[i] = MakeNode(NeuronType.Input, activation, NeuronDirection.Forward);
        }

        for (var i = 0; i < outputSize; i++)
        {
            _outputs[i] = MakeNode(NeuronType.Output, activation, NeuronDirection.Forward);
        }

        foreach (var input in _inputs)
        {
            foreach (var output in _outputs)
            {
                var weight = Random.NextDouble() * 2 - 1;
                MakeEdge(input, output, (float) weight);
            }
        }
    }

    public Neat(ModelWrap wrap)
    {
        var neat = wrap.NeatWrap;
        _inputs = neat.Inputs
            .Select(input => new NeuronId { Index = input.Index })
            .ToArray();
        _outputs = neat.Outputs
            .Select(output => new NeuronId { Index = output.Index })
            .ToArray();
        _nodes = neat.Nodes
            .Select(node => node)
            .ToList();
        _activation = neat.Activation;
        _edges = neat.Edges
            .Select(edge => new Edge(edge))
            .ToList();
        _edgeInnovationLookup = neat.EdgeInnovationLookup
            .Select(pair => (Id: pair.Key, edge: new EdgeId { Index = pair.Value.Index }))
            .ToDictionary(key => key.Id, val => val.edge);
    }
    
    private Neat(Neat neat) : base(neat.InnovationId)
    {
        _inputs = neat._inputs
            .Select(input => new NeuronId { Index = input.Index })
            .ToArray();
        _outputs = neat._outputs
            .Select(output => new NeuronId { Index = output.Index })
            .ToArray();
        _nodes = neat._nodes
            .Select(node => node.Clone())
            .ToList();
        _activation = neat._activation;
        _edges = neat._edges
            .Select(edge => new Edge(edge))
            .ToList();
        _edgeInnovationLookup = neat._edgeInnovationLookup
            .Select(pair => (Id: pair.Key, edge: new EdgeId { Index = pair.Value.Index }))
            .ToDictionary(key => key.Id, val => val.edge);
    }
    
    public List<float> GetOutputs() => 
        _outputs
            .Select(output => _nodes[output.Index].ActivatedValue)
            .ToList();
    
    private void AddNode(Activation activation, NeuronDirection direction)
    {
        var newNodeId = MakeNode(NeuronType.Hidden, activation, direction);
        var currentEdge = _edges[Random.Next(_edges.Count)];

        MakeEdge(currentEdge.Src, newNodeId, 1.0f);
        MakeEdge(newNodeId, currentEdge.Dst, currentEdge.Weight);
        
        _edges[currentEdge.Id.Index].Disable(_nodes);
    }
    
    private void AddEdge()
    {
        var sending = RandomNodeNotOfType(NeuronType.Output);
        var receiving = RandomNodeNotOfType(NeuronType.Input);

        if (ValidConnection(sending, receiving))
        {
            MakeEdge(sending, receiving, (float) Random.NextDouble());
        }
    }

    private NeuronId MakeNode(NeuronType neuronType, Activation activation, NeuronDirection direction)
    {
        var nodeId = new NeuronId { Index = _nodes.Count };
        _nodes.Add(new Neuron(nodeId, neuronType, activation, direction, (float) Random.NextDouble()));
        return nodeId;
    }

    private EdgeId MakeEdge(NeuronId src, NeuronId dst, float weight)
    {
        var edgeId = new EdgeId { Index = _edges.Count };
        var edge = new Edge(edgeId, src, dst, weight, true);
        
        edge.LinkNodes(_nodes);
        _edgeInnovationLookup[edge.Innovation] = edgeId;
        _edges.Add(edge);

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
        var receivingNode = _nodes[receiving.Index];
        var stack = receivingNode.OutgoingEdges()
            .Select(edge => _edges[edge.Index].Dst.Index)
            .ToList();

        while (stack.Any())
        {
            var topNode = stack.Last();
            
            stack.RemoveAt(stack.Count - 1);

            var current = _nodes[topNode];
            if (current.Id.Equals(sending))
            {
                return true;
            }

            stack.AddRange(current.OutgoingEdges().Select(edge => _edges[edge.Index].Dst.Index));
        }

        return false;
    }

    private bool Exists(NeuronId sending, NeuronId receiving) => 
        _edges.Any(edge => edge.Src.Equals(sending) && edge.Dst.Equals(receiving));


    private NeuronId RandomNodeNotOfType(NeuronType neuronType)
    {
        var node = _nodes[Random.Next(_nodes.Count)];

        while (node.NeuronType == neuronType)
        {
            node = _nodes[Random.Next(_nodes.Count)];
        }

        return node.Id;
    }


    private void EditWeights(float editable, float size)
    {
        foreach (var edge in _edges)
        {
            var shouldEdit = Random.NextDouble() < editable;
            var weightValue = shouldEdit 
                ? (float) Random.NextDouble() 
                : edge.Weight * ((float) Random.NextDouble() * size - size);
            edge.UpdateWeight(weightValue, _nodes);
        }

        foreach (var node in _nodes)
        {
            var shouldEdit = Random.NextDouble() < editable;
            node.Bias = shouldEdit ? (float) Random.NextDouble() : node.Bias * ((float) Random.NextDouble() * size - size);
        }
    }

    public float[] Forward(float[] data)
    {
        if (_inputs.Length != data.Length)
        {
            throw new Exception($"Dense layer input does not match. Input size {_inputs.Length}, given size {data.Length}");
        }
        
        var outputs = new List<float>();
        var updates = new List<NodeUpdate>();
        var pendingCount = 0;
        var lowestPendingIdx = _nodes.Count;

        var inputCounter = 0;
        foreach (var node in _nodes)
        {
            node.Prepare();

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
            var endIdx = _nodes.Count;
            lowestPendingIdx = endIdx;

            for (var i = startIdx; i < endIdx; i++)
            {
                var node = _nodes[i];
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
        
        return outputs.ToArray();
    }
    
    public ModelWrap Save() => new ModelWrap
    {
        ModelType = ModelType.Neat,
        NeatWrap = new()
        {
            Inputs = _inputs
                .Select(input => new NeuronId { Index = input.Index })
                .ToArray(),
            Outputs = _outputs
                .Select(output => new NeuronId { Index = output.Index })
                .ToArray(),
            Nodes = _nodes
                .Select(node => node)
                .ToList(),
            Edges = _edges
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
            Activation = _activation,
            EdgeInnovationLookup = _edgeInnovationLookup
                .Select(pair => (Id: pair.Key, edge: new EdgeId { Index = pair.Value.Index }))
                .ToDictionary(key => key.Id, val => val.edge),   
        }
    };
    
    public T Crossover<T, TE>(T other, TE environment, double crossoverRate)         
        where T: class, IGenome
        where TE: EvolutionEnvironment
    {
        var child = CloneGenome<Neat>();
        var parentTwo = other as Neat;
        var neatEnv = environment as NeatEnvironment;

        if (Random.NextDouble() < crossoverRate)
        {
            foreach (var edge in child._edges)
            {
                if (parentTwo._edgeInnovationLookup.ContainsKey(edge.Innovation))
                {
                    var parentEdge = parentTwo._edges.Single(pedge => pedge.Innovation == edge.Innovation);

                    if (Random.NextDouble() < .5)
                    {
                        edge.UpdateWeight(parentEdge.Weight, child._nodes);
                    }

                    if ((!edge.Active || !parentEdge.Active) && Random.NextDouble() < neatEnv.ReactivateRate)
                    {
                        edge.Enable(child._nodes);
                    }
                }
            }
        }
        else
        {
            if (Random.NextDouble() < neatEnv.WeightMutateRate)
            {
                child.EditWeights(neatEnv.EditWeights, neatEnv.WeightPerturb);
            }

            if (Random.NextDouble() < neatEnv.NewNodeRate)
            {
                var actFunction = neatEnv.ActivationFunctions[Random.Next(neatEnv.ActivationFunctions.Count)];
                if (Random.NextDouble() < neatEnv.RecurrentNeuronRate)
                {
                    child.AddNode(actFunction, NeuronDirection.Recurrent);
                }
                else
                {
                    child.AddNode(actFunction, NeuronDirection.Forward);
                }
            }

            if (Random.NextDouble() < neatEnv.NewEdgeRate)
            {
                child.AddEdge();
            }
        }

        return child as T;
    }


    public async Task<double> Distance<T>(T other, PopulationControl populationControl)
    {
        var parentTwo = other as Neat;
        
        var parentOneLookup = _edges.ToDictionary(key => key.InnovationId, val => val.Weight);
        var parentTwoLookup = parentTwo._edges.ToDictionary(key => key.InnovationId, val => val.Weight);
        
        return await DistanceCalculator.Distance(parentOneLookup, parentTwoLookup, populationControl);
    }

    public T CloneGenome<T>() where T : class => new Neat(this) as T;

    public void ResetGenome()
    {
        foreach (var node in _nodes)
        {
            node.Reset();
        }
    }
    
    public Prediction Predict(Tensor input)
    {
        var output = Forward(input.ToArray());
        var maxIndex = output.ToList().IndexOf(output.Max());

        return new Prediction(output.ToTensor(), maxIndex, output[maxIndex]);
    }
}
