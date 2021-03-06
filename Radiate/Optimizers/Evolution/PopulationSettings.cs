
namespace Radiate.Optimizers.Evolution;

public class PopulationSettings
{
    public PopulationSettings() { }
    
    public int? Size { get; set; }
    public bool DynamicDistance { get; set; } = true;
    public double SpeciesDistance { get; set; } = .5;
    public double InbreedRate { get; set; } = .001;
    public double CrossoverRate { get; set; } = .75;
    public int StagnationLimit { get; set; } = 15;
    public int SpeciesTarget { get; set; } = 5;
    public double COne { get; set; } = 1.0;
    public double CTwo { get; set; } = 1.0;
    public double CThree { get; set; } = 0.4;
}
