namespace Radiate.Records;

public record PopulationControl(double Distance, 
    double DistancePrecision = 0.1,
    int StagnationCount = 0, 
    double PreviousFitness = 0);