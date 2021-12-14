using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Radiate.Net.Examples.Examples;

namespace Radiate.Net.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            var options = "\n[0] Stop\n" +
                          "[1] Evolve Dense\n" +
                          "[2] Evolve Hello World\n" +
                          "[3] Train MLP Dense\n" +
                          "[4] Train MLP LSTM\n" +
                          "[5] Boston Housing\n" +
                          "[6] Dense Neural Net Minst\n" +
                          "[7] Conv Net Minst\n" +
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
                await new EvolveDense().Run();
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
                await new NeuralNetDenseMinst().Run();
            }

            if (choice == 7)
            {
                await new ConvNetMinst().Run();
            }
        }
    }
}