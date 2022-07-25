using Radiate.Optimizers.Evolution;

namespace Radiate.Callbacks.Interfaces;

public interface IGenerationEvolvedCallback : ITrainingCallback
{
    void GenerationEvolved(object sender, GenerationEvolved generationEvolved);
}

public class GenerationEvolved : EventArgs
{
    public int GenerationNum { get; set; }
    public GenerationReport Report { get; set; }
}