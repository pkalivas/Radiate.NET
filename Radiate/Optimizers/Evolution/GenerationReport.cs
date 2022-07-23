namespace Radiate.Optimizers.Evolution;

public class GenerationReport
{
    public int NumMembers { get; set; }
    public int NumNiche { get; set; }
    public float TopFitness { get; set; }
    public int StagnationCount { get; set; }
    public double Distance { get; set; }
    public List<NicheReport> NicheReports { get; set; }
}

public class NicheReport
{
    public int Innovation { get; set; }
    public Guid Id { get; set; }
    public Guid Mascot { get; set; }
    public int Age { get; set; }
    public double AdjustedFitness { get; set; }
    public int NumMembers { get; set; }
}