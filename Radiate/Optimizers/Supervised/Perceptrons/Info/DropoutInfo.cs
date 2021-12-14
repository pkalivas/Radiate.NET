namespace Radiate.Optimizers.Supervised.Perceptrons.Info
{
    public class DropoutInfo : LayerInfo
    {
        private const float DefaultDropoutPercent = 0.2f;
        
        public float DropoutPercent { get; set; }

        public DropoutInfo() : this(DefaultDropoutPercent) { }
        
        public DropoutInfo(float dropoutPercent)
        {
            DropoutPercent = dropoutPercent;
        }
    }
}