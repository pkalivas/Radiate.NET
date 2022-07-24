
namespace Radiate.Optimizers.Evolution;

public delegate float FitnessFunction<in T>(T model) where T : class;
