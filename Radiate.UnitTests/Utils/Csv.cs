using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.UnitTests.Utils;

public static class Csv
{
    public static async Task<Tensor> LoadTensor(string fileName)
    {
        var contents = await File.ReadAllTextAsync(fileName);

        var values = new List<float>();
        foreach (var row in contents.Split("\n").Skip(1))
        {
            var val = row
                .Split(",")
                .Skip(1)
                .Select(Convert.ToSingle)
                .ToList();
            
            values.AddRange(val);
        }

        var nameWithoutFileExt = fileName.Split("\\").Last().Split(".").First();
        var resultShapeSplit = nameWithoutFileExt.Split("_").Skip(1).Select(val => Convert.ToInt32(val)).ToArray();
        var resultShape = resultShapeSplit.Length switch
        {
            1 => new Shape(resultShapeSplit[0]),
            2 => new Shape(resultShapeSplit[0], resultShapeSplit[1]),
            3 => new Shape(resultShapeSplit[0], resultShapeSplit[1], resultShapeSplit[2])
        };

        return values.ToTensor().Reshape(resultShape);
    }
    
    public static async Task<List<Tensor>> LoadFromCsv(string baseDirectory, string tensorType)
    {
        var result = new List<Tensor>();
        var directory = $"{Environment.CurrentDirectory}\\{baseDirectory}";

        var fileNames = Directory.GetFiles(directory).Where(file => file.Contains(tensorType)).ToList();

        var iter = fileNames.OrderBy(file =>
            {
                var temp = file.Split("\\").Last().Split("-").ToList();
                if (temp.Count > 1)
                {
                    return Convert.ToInt32(temp.First());
                }

                return 0;
            })
            .ToList();
        
        foreach (var name in iter)
        {
            result.Add(await Csv.LoadTensor(name));    
        }

        return result;
    }
}
