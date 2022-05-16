
namespace Radiate.Optimizers.Evolution.Environment;

public delegate float Solve<in T>(T model) where T : class;
