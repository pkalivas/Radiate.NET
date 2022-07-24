namespace Radiate.Optimizers.Evolution;

public class Allele
{
    public int InnovationId { get; }
    protected Random Random { get; }

    protected Allele(int innovationId)
    {
        InnovationId = innovationId;
        Random = RandomGenerator.RandomGenerator.Seed is null
            ? RandomGenerator.RandomGenerator.Next
            : new Random(InnovationId);
    }

    protected Allele()
    {
        var count = InnovationCounter.Increment();

        InnovationId = count;
        Random = RandomGenerator.RandomGenerator.Seed is null
            ? RandomGenerator.RandomGenerator.Next
            : new Random(count);
    }
}