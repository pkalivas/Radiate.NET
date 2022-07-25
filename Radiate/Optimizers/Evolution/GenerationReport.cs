namespace Radiate.Optimizers.Evolution;

public class GenerationReport
{
    public int NumMembers { get; set; }
    public float TopFitness { get; set; }
    public SpeciesReport SpeciesReport { get; init; }
}

public class SpeciesReport
{
    public double Distance { get; init; }
    public Dictionary<Guid, int> SpeciesStagnation { get; init; }
    public List<NicheReport> NicheReports { get; init; }
}

public class NicheReport
{
    public int Innovation { get; set; }
    public Guid SpeciesId { get; set; }
    public Guid MascotId { get; set; }
    public int Age { get; set; }
    public double AdjustedFitness { get; set; }
    public double MaxFitness { get; init; }
    public double MinFitness { get; init; }
    public int NumMembers { get; set; }
}