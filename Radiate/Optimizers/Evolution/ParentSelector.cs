namespace Radiate.Optimizers.Evolution;

public static class ParentSelector
{
    public static (Guid parentOne, Guid parentTwo) Select(double inbreedRate, ICollection<Species> species)
    {
        var random = RandomGenerator.RandomGenerator.Next;
        if (random.NextDouble() < inbreedRate)
        {
            var niche = GetBiasedRandomNiche(species, random);
            var parentOne = GetBiasedRandomMember(niche, random);
            var parentTwo = GetBiasedRandomMember(niche, random);

            return (parentOne, parentTwo);
        }
        else
        {
            var speciesOne = GetBiasedRandomNiche(species, random);
            var speciesTwo = GetBiasedRandomNiche(species, random);

            var parentOne = GetBiasedRandomMember(speciesOne, random);
            var parentTwo = GetBiasedRandomMember(speciesTwo, random);

            return (parentOne, parentTwo);
        }
    }
    
    private static Species GetBiasedRandomNiche(ICollection<Species> species, Random random)
    {
        var total = species
            .Aggregate(0.0, (all, current) => all + current.Fitness);
        
        var runningTotal = 0.0;
        var index = random.NextDouble() * total;
        
        foreach (var niche in species)
        {
            runningTotal += niche.Fitness;
            if (runningTotal >= index)
            {
                return niche;
            }
        }

        return species.First();
    }


    private static Guid GetBiasedRandomMember(Species species, Random random)
    {
        if (random.NextDouble() < .5)
        {
            return species.BestMember();
        }

        var speciesMembers = species.Members;
        var index = species.Fitness * random.NextDouble();
        var runningTotal = 0.0;

        foreach (var (id, fitness) in speciesMembers)
        {
            runningTotal += fitness;
            if (runningTotal >= index)
            {
                return id;
            }
        }
        
        return speciesMembers.First().GenomeId;
    }
}