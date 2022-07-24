namespace Radiate.Records;

public record DistanceControl(double COne, double CTwo, double CThree);

public record StagnationControl(int StagnationLimit, int StagnationCount = 0, double PreviousFitness = 0);

public record PassDownControl(double InbreedRate, double CrossoverRate, int Size);
