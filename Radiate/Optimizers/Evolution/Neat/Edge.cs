
namespace Radiate.Optimizers.Evolution.Neat;

public class Edge : Allele
{
    public EdgeId Id { get; set; }
    public Guid Innovation { get; set; }
    public NeuronId Src { get; set; }
    public NeuronId Dst { get; set; }
    public float Weight { get; set; }
    public bool Active { get; set; }

    public Edge() { }

    public Edge(EdgeId id, NeuronId src, NeuronId dst, float weight, bool active) : base(id.Index)
    {
        Id = id;
        Innovation = Guid.NewGuid();
        Src = src;
        Dst = dst;
        Weight = weight;
        Active = active;
    }

    public Edge(Edge edge) : base(edge.InnovationId)
    {
        Id = new EdgeId { Index = edge.Id.Index };
        Innovation = edge.Innovation;
        Src = new NeuronId { Index = edge.Src.Index };
        Dst = new NeuronId { Index = edge.Dst.Index };
        Weight = edge.Weight;
        Active = edge.Active;
    }

    public void Update(float delta, List<Neuron> nodes)
    {
        UpdateWeight(Weight + delta, nodes);
    }

    public double Calculate(float value) => value * Weight;
    
    public void UpdateWeight(float weight, List<Neuron> nodes)
    {
        Weight = weight;
        nodes[Dst.Index].UpdateIncoming(this, weight);
    }

    public void LinkNodes(List<Neuron> nodes)
    {
        nodes[Src.Index].AddOutgoing(Id);
        nodes[Dst.Index].AddIncoming(this);
    }

    public void Enable(List<Neuron> nodes)
    {
        if (Active)
        {
            return;
        }

        Active = true;
        nodes[Src.Index].AddOutgoing(Id);
        nodes[Dst.Index].UpdateIncoming(this, Weight);
    }

    public void Disable(List<Neuron> nodes)
    {
        Active = false;
        nodes[Src.Index].RemoveOutgoing(Id);
        nodes[Dst.Index].UpdateIncoming(this, 0.0f);
    }

}
