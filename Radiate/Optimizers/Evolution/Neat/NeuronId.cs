
namespace Radiate.Optimizers.Evolution.Neat
{
    public class NeuronId
    {
        public int Index { get; set; }

        public override bool Equals(object? obj)
        {
            if ((obj == null) || this.GetType() != obj.GetType())
            {
                return false;
            }

            var edge = (NeuronId)obj;
            return edge.Index == Index;
        }

        public override int GetHashCode()
        {
            return Index.GetHashCode();
        }
    }
}