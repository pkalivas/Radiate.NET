namespace Radiate.Domain.Extensions;

public static class ListExtensions
{
    private const float Tolerance = 0.001f;
    
    public static List<float[]> Normalize(this List<float[]> data)
    {
        var minLookup = new Dictionary<int, float>();
        var maxLookup = new Dictionary<int, float>();
        
        var featureLength = data
            .Select(row => row.Length)
            .Distinct()
            .Single();
        
        foreach (var index in Enumerable.Range(0, featureLength))
        {
            var column = data.Select(point => point[index]).ToList();
            minLookup[index] = column.Min();
            maxLookup[index] = column.Max();
        }

        return data
            .Select(row => row
                .Select((feature, index) =>
                {
                    var min = minLookup[index];
                    var max = maxLookup[index];

                    var denominator = Math.Abs(min - max) < Tolerance ? 1 : max - min;
                    return (feature / (min == 0 ? 1 : min)) / denominator;
                })
                .ToArray())
            .ToList();
    }
    
    public static List<float[]> Standardize(this List<float[]> data)
    {
        var meanLookup = new Dictionary<int, float>();
        var stdLookup = new Dictionary<int, float>();
        
        var featureLength = data
            .Select(row => row.Length)
            .Distinct()
            .Single();
        
        foreach (var index in Enumerable.Range(0, featureLength))
        {
            var column = data.Select(point => point[index]).ToList();
            var average = column.Average();
            var sum = column.Sum(val => (float) Math.Pow(val - average, 2));
            
            meanLookup[index] = column.Average();
            stdLookup[index] = (float)Math.Sqrt(sum / (column.Count - 1));
        }
        
        return data
            .Select(row => row
                .Select((feature, index) => (feature - meanLookup[index]) / stdLookup[index])
                .ToArray())
            .ToList();
    }
    
    public static List<float[]> OneHotEncode(this List<float[]> targets)
    {
        var targetCount = targets.SelectMany(row => row).Distinct().Count();
        return targets
            .Select(tar => Enumerable
                .Range(0, targetCount)
                .Select((_, index) => Math.Abs(index - tar.First()) < Tolerance ? 1f : 0.0f)
                .ToArray())
            .ToList();
    }

}