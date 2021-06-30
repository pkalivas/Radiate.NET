using System;
using System.Collections.Generic;
using System.Linq;
using Radiate.NET.Environments.MarketEnvironment.Strategies;
using Radiate.NET.Models;

namespace Radiate.NET.Environments.MarketEnvironment
{
    public class Market
    {
        public float StopLossPct { get; set; }
        public float TakeProfitPct { get; set; }
        public float MaxCash { get; set; }
        public float StartingCash { get; set; }

        public List<List<float>> Data { get; set; }
        public List<float> Prices { get; set; }
        public List<DateTime> Dates { get; set; }
        public IStrategy Strategy { get; set; }
        

        public float AnalyzeModel<T>(T model) where T : ITradingModel
        {
            var portfolio = CreatePortfolio();

            foreach (var (row, idx) in Data.Select((row, idx) => (row, idx)))
            {
                var currentPrice = Prices[idx];
                var currentDate = Dates[idx];

                var action = GetAction(model.Forward(row));

                portfolio.Step(idx, currentPrice, currentDate, action);
            }

            return Strategy.GetReward(portfolio);
        }


        private static string GetAction(List<float> outputs)
        {
            var bestIdx = 0;
            var bestVal = 0f;

            foreach (var (choice, idx) in outputs.Select((val, idx) => (val, idx)))
            {
                if (choice > bestVal)
                {
                    bestIdx = idx;
                    bestVal = choice;
                }
            }

            return bestIdx switch
            {
                0 => "Buy",
                1 => "Sell",
                2 => "Hold",
                _ => throw new Exception($"Output had more than three choices ")
            };
        }

        private Portfolio CreatePortfolio() => new()
        {
            StartingValue = StartingCash,
            CurrentValue = StartingCash,
            CurrentCash = StartingCash,
            TakeProfitPct = TakeProfitPct,
            StopLossPct = StopLossPct,
            MaxCash = MaxCash,
            Broke = false,
            State = new State(),
            OpenPositions = new List<Position>(),
            ClosedPositions = new List<Position>(),
            Predictions = new List<Prediction>()
        };
    }
}