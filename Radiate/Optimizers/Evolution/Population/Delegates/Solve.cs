
namespace Radiate.Optimizers.Evolution.Population.Delegates;

public delegate float Solve<in T>(T model) where T : Genome;
