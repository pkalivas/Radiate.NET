
namespace Radiate.Data.Utils;

public static class MinstDiscriptor
{
    public static void Describe(List<float[]> inputs, List<float[]> features)
    {
        Console.WriteLine($"{string.Join("-", Enumerable.Range(0, 50).Select(_ => ""))}");
        Console.WriteLine($"Total data points: {inputs.Count}");
        Console.WriteLine($"Label occurances:");
        
        foreach (var group in features.GroupBy(val => val[0]).OrderBy(val => val.Key))
        {
            Console.WriteLine($"\t{group.Key} occurs {group.ToList().Count} times.");
        }
    }
}