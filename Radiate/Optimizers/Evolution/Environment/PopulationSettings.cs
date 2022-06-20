
namespace Radiate.Optimizers.Evolution.Environment;

public class PopulationSettings
{
    public PopulationSettings() { }
    
    public PopulationSettings(int size)
    {
        Size = size;
    }
    
    public int Size { get; set; }
    public bool DynamicDistance { get; set; } = true;
    public double SpeciesDistance { get; set; } = .5;
    public double InbreedRate { get; set; } = .001;
    public double CrossoverRate { get; set; } = .75;
    public int StagnationLimit { get; set; } = 15;
    public double CleanPct { get; set; } = .9;
    public int SpeciesTarget { get; set; } = 5;
}
