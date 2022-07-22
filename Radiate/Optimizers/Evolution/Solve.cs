
namespace Radiate.Optimizers.Evolution;

public delegate float Solve<in T>(T model) where T : class;
