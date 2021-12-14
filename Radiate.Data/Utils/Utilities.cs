using System.IO;
using System.IO.Compression;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace Radiate.Data.Utils
{
    public static class Utilities
    {
        public static async Task<T> UnzipGZAndLoad<T>(string filePath)
        {
            await using var fileStream = File.OpenRead(filePath);
            await using var zippedStream = new GZipStream(fileStream, CompressionMode.Decompress);
            
            using var jsonReader = new JsonTextReader(new StreamReader(zippedStream));

            return JsonSerializer.CreateDefault().Deserialize<T>(jsonReader);
        }
    }
}