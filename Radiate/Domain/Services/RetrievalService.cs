using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace Radiate.Domain.Services
{
    public class RetrievalService
    {
        private const int CsvHeaderRows = 1;
        
        public static async Task<string> FetchAndSave(Uri fetchUrl, string downloadDirectory, string fileName)
        {
            var fullPath = $"{downloadDirectory}\\{fileName}";
            if (!Directory.Exists(downloadDirectory))
            {
                Directory.CreateDirectory(downloadDirectory);
            }

            var client = new HttpClient();

            var trainResponse = await client.GetAsync(fetchUrl);
            var trainStream = await trainResponse.Content.ReadAsStreamAsync();

            var trainFile = File.OpenWrite(fullPath);
            await trainStream.CopyToAsync(trainFile);

            trainFile.Close();

            return fullPath;
        }
        
    }
}