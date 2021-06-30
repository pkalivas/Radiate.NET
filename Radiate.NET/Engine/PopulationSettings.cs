
namespace Radiate.NET.Engine
{
    public class PopulationSettings
    {
        public int Size { get; set; } = 100;
        public bool DynamicDistance { get; set; } = true;
        public double SpeciesDistance { get; set; }
        public double InbreedRate { get; set; }
        public double CrossoverRate { get; set; }
        public int StagnationLimit { get; set; }
        public double CleanPct { get; set; }
        public int SpeciesTarget { get; set; }
        public bool UseLogger { get; set; } = false;
    }
}