namespace Radiate.Records;

public record DistanceControl(double COne, double CTwo, double CThree);

public record StagnationControl(int StagnationLimit, int StagnationCount = 0, double MaxFitness = 0);

public record PassDownControl(double InbreedRate, double CrossoverRate, int Size);

public record CompatibilityControl(bool Dynamic, int Target, double Distance);
