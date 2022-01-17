
namespace Radiate.Optimizers.Evolution.Population;

public delegate float Solve<in T>(T model) where T : Genome;
