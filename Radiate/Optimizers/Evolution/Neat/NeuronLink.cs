using System;

namespace Radiate.Optimizers.Evolution.Neat
{
    public class NeuronLink
    {
        public EdgeId Id { get; set; }
        public NeuronId Src { get; set; }
        public float Weight { get; set; }

        public override bool Equals(object obj)
        {
            if ((obj == null) || this.GetType() != obj.GetType())
            {
                return false;
            }

            var link = (NeuronLink)obj;
            return link.Id.Equals(Id) && link.Src.Equals(Src);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Id, Src, Weight);
        }
    }
}