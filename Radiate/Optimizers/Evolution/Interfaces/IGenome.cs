﻿using Radiate.Records;

namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IGenome : IOptimizerModel
{
    public T Crossover<T, TE>(T other, TE environment, double crossoverRate)
        where T: class, IGenome
        where TE: EvolutionEnvironment;

    public Task<double> Distance<T>(T other, PopulationControl populationControl);

    public T CloneGenome<T>() where T : class;

    public void ResetGenome();
}