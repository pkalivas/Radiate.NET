using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public static class DistanceCalculator
{
    public static double Distance(IReadOnlyDictionary<int, float> parentOne, 
        IReadOnlyDictionary<int, float> parentTwo, 
        DistanceControl distanceControl)
    {
        var (largestParent, smallestParent) = parentOne.Count > parentTwo.Count 
            ? (parentOne, parentTwo) 
            : (parentTwo, parentOne);
        var largestCount = (double) largestParent.Count;

        if (largestCount < EvolutionConstants.InnovationMax)
        {
            largestCount = 1.0;
        }

        var (c1, c2, c3) = distanceControl;
        
        var excessGenes = parentOne.Count > parentTwo.Count
            ? parentOne.Count - parentTwo.Count
            : parentTwo.Count - parentOne.Count;
        
        var (oneDiff, oneDisjoint) = GetValues(largestParent, smallestParent);
        var (_, twoDisjoint) = GetValues(smallestParent, largestParent);
        var totalDisjoint = oneDisjoint + twoDisjoint;
        
        var e = (c1 * excessGenes) / largestCount;
        var d = (c2 * totalDisjoint) / largestCount;
        var v = c3 * oneDiff;

        return e + d + v;
    }

    private static (double AvgWeightDiff, double Disjoint) GetValues(IReadOnlyDictionary<int, float> one, IReadOnlyDictionary<int, float> two)
    {
        var diff = 0.0;
        var disjoint = 0.0;
        var shared = 0.0;

        foreach (var (innov, weight) in one)
        {
            if (two.ContainsKey(innov))
            {
                diff += weight;
                shared += 1.0;
            }
            else
            {
                disjoint += 1.0;
            }
        }

        var weightDiff = shared == 0.0 ? 1.0 : diff / shared;
        var disjointGenes = disjoint == 0.0 ? 1.0 : disjoint;

        return (weightDiff, disjointGenes);
    }
}