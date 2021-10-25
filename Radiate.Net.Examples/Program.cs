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
                          "[2] Evolve LSTM\n" +
                          "[3] Evolve RNN\n" +
                          "[4] Train Dense\n" +
                          "[5] Train LSTM\n" +
                          "[6] Train RNN\n" +
                          "[7] Temperature Time Series\n" +
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
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            
            if (choice == 1)
            {
                await new EvolveDense().Run();
            }

            if (choice == 2)
            {
                await new EvolveLSTM().Run();
            }

            if (choice == 3)
            {
                await new EvolveRNN().Run();
            }

            if (choice == 4)
            {
                await new TrainDense().Run();
            }

            if (choice == 5)
            {
                await new TrainLSTM().Run();
            }

            if (choice == 6)
            {
                await new TrainRNN().Run();
            }

            if (choice == 7)
            {
                await new TemperatureTimeSeries().Run();
            }
            
            stopwatch.Stop();
            Console.WriteLine($"Finished in {stopwatch.ElapsedMilliseconds} milliseconds.");
        }
    }
}