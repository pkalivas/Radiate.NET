
namespace Radiate.Optimizers.Evolution.Population.Delegates;

public delegate double Solve<in T>(T model) where T : Genome;
