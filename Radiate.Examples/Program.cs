using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Radiate.Examples.Examples;

namespace Radiate.Examples;

class Program
{
    static void Main(string[] args)
    {
        RandomGenerator.RandomGenerator.Seed = null;

        var serviceProvider = new ServiceCollection()
            .AddLogging(builder => builder.AddConsole())
            .AddServices()
            .BuildServiceProvider();

        var options = "\n[0] Stop\n" +
                      "[1] Evolve NEAT\n" +
                      "[2] Evolve Hello World\n" +
                      "[3] Simple MLP\n" +
                      "[4] LSTM\n" +
                      "[5] Regression\n" +
                      "[6] Conv Net\n" +
                      "[7] KMeans\n" + 
                      "[8] RandomForest\n" + 
                      "[9] SupportVectorMachine\n" +
                      "[10] Evolve Tree\n" +
                      "[11] Evolve Forest\n" +
                      "[12] Temp Time Series\n" +
                      ":: ";

        while (true)
        {
            Console.Write(options);
            var choice = Convert.ToInt32(Console.ReadLine());
        
            if (choice == 0)
            {
                break;
            }

            Dispatch(choice, serviceProvider).ConfigureAwait(true).GetAwaiter().GetResult();
        }
    }

    private static async Task Dispatch(int choice, IServiceProvider serviceProvider)
    {
        await using var scope = serviceProvider.CreateAsyncScope();
        var exampleResolver = scope.ServiceProvider.GetRequiredService<ExampleResolver>();

        await exampleResolver(choice).Run();
    }
    
}