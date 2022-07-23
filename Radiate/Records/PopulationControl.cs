namespace Radiate.Records;

public record PopulationControl(bool DynamicDistance, 
    double Distance, 
    int SpeciesTarget,
    double COne,
    double CTwo,
    double CThree,
    double DistancePrecision = 0.1);

public record StagnationControl(double CleanPercent, int StagnationLimit, int StagnationCount = 0, double PreviousFitness = 0);

public record PassDownControl(double InbreedRate, double CrossoverRate, int Size);
