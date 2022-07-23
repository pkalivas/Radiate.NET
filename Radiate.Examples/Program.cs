using Radiate.Examples.Examples;

namespace Radiate.Examples;

class Program
{
    static void Main(string[] args)
    {
        RandomGenerator.RandomGenerator.Seed = null;
        
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
                      ":: ";

        while (true)
        {
            Console.Write(options);
            var choice = Convert.ToInt32(Console.ReadLine());

            if (choice == 0)
            {
                break;
            }
            
            Dispatch(choice).ConfigureAwait(true).GetAwaiter().GetResult();
        }
    }

    private static async Task Dispatch(int choice)
    {
        if (choice == 1)
        {
            await new EvolveNEAT().Run();
        }

        if (choice == 2)
        {
            await new EvolveHelloWorld().Run();
        }

        if (choice == 3)
        {
            await new TrainDense().Run();
        }

        if (choice == 4)
        {
            await new TrainLSTM().Run();
        }

        if (choice == 5)
        {
            await new BostonRegression().Run();
        }

        if (choice == 6)
        {
            await new ConvNetMinst().Run();
        }

        if (choice == 7)
        {
            await new BlobMeans().Run();
        }

        if (choice == 8)
        {
            await new RandomForestPredictor().Run();
        }

        if (choice == 9)
        {
            await new SVMPredictor().Run();
        }

        if (choice == 10)
        {
            await new EvolveTree().Run();
        }

        if (choice == 11)
        {
            await new EvolveForest().Run();
        }
    }
}