namespace Radiate.Optimizers.Evolution.Environment;

public static class ParentSelector
{
    public static (Guid parentOne, Guid parentTwo) Select(double inbreedRate, List<Niche> species)
    {
        var random = new Random();

        Guid parentOne;
        Guid parentTwo;

        if (random.NextDouble() < inbreedRate)
        {
            var niche = GetBiasedRandomNiche(species, random);
            parentOne = GetBiasedRandomMember(niche, random);
            parentTwo = GetBiasedRandomMember(niche, random);
        }
        else
        {
            var speciesOne = GetBiasedRandomNiche(species, random);
            var speciesTwo = GetBiasedRandomNiche(species, random);

            parentOne = GetBiasedRandomMember(speciesOne, random);
            parentTwo = GetBiasedRandomMember(speciesTwo, random);
        }

        return (parentOne, parentTwo);
    }
    
    private static Niche GetBiasedRandomNiche(List<Niche> species, Random random)
    {
        var total = species.Aggregate(0.0, (all, current) => all + current.TotalAdjustedFitness);
        
        var runningTotal = 0.0;
        var index = random.NextDouble() * total;
        
        foreach (var niche in species)
        {
            runningTotal += niche.TotalAdjustedFitness;
            if (runningTotal >= index)
            {
                return niche;
            }
        }

        return species.First();
    }


    private static Guid GetBiasedRandomMember(Niche species, Random random)
    {
        var index = species.TotalAdjustedFitness * random.NextDouble();
        var runningTotal = 0.0;

        foreach (var (id, fitness) in species.Members)
        {
            runningTotal += fitness;
            if (runningTotal >= index)
            {
                return id;
            }
        }

        return species.Members.First().memberId;
    }
}