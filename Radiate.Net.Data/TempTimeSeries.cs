using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace Radiate.Net.Data
{
    internal class TempPoint
    {
        public DateTime Date { get; set; }
        public double Temp { get; set; }
    }
    
    public class TempTimeSeries : IDataSet
    {
        public (List<List<float>> inputs, List<List<float>> targets) GetDataSet()
        {
            var file = File.ReadAllText(Directory.GetCurrentDirectory()+ "\\DataSets\\TempTimeSeries.json");
            var temps = JsonConvert.DeserializeObject<List<TempPoint>>(file)!.Take(100).ToList();

            var normalizedData = Normalize(temps);

            var inputs = new List<List<float>>();
            var targets = new List<List<float>>();

            for (var i = 0; i < normalizedData.Count - 2; i++)
            {
                inputs.Add(new() { normalizedData[i] });
                targets.Add(new() { normalizedData[i + 1] });
            }

            return (inputs, targets);
        }

        private static List<float> Normalize(List<TempPoint> points)
        {
            var births = points.Select(birth => birth.Temp).ToList();
            var min = (float)births.Min();
            var max = (float)births.Max();

            return births
                .Select(point => (float) ((point - min) / (max - min)))
                .ToList();
        }
    }
}