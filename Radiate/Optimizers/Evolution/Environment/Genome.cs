﻿
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Environment;

public interface IGenome
{
    public T Crossover<T, TE>(T other, TE environment, double crossoverRate)
        where T: class, IGenome
        where TE: EvolutionEnvironment;

    public Task<double> Distance<T, TE>(T other, TE environment);

    public T CloneGenome<T>() where T : class;

    public void ResetGenome();
    public T Randomize<T>() where T : class;
}