using System;
using System.Collections.Generic;
using System.Linq;

namespace Radiate.NET.Environments.MarketEnvironment
{
    public class State
    {
        public float CurrentPrice { get; set; }
        public int WinningTrades { get; set; }
        public int LosingTrades { get; set; }
    }

    public class Portfolio
    {
        public float StartingValue { get; set; }
        public float CurrentValue { get; set; }
        public float CurrentCash { get; set; }
        public float TakeProfitPct { get; set; }
        public float StopLossPct { get; set; }
        public float MaxCash { get; set; }
        public bool Broke { get; set; } 
        public State State { get; set; }
        public List<Position> OpenPositions { get; set; }
        public List<Position> ClosedPositions { get; set; }
        public List<Prediction> Predictions { get; set; }

        public bool IsValid() => !Broke && ClosedPositions.Any();

        public float MarketReturn()
        {
            var firstValue = Predictions.First().Price;
            var lastValue = Predictions.Last().Price;

            return ((lastValue / firstValue) - 1) * 100;
        }

        public float UnrealizedGains() => 
            OpenPositions
                .Select(pos => (pos.SharesBought* State.CurrentPrice) - pos.OpenCash)
                .Sum();

        public float RealizedGains() =>
            ClosedPositions
                .Select(pos => pos.CloseCash - pos.OpenCash)
                .Sum();

        public float Gains() => StartingValue + UnrealizedGains() + RealizedGains();

        public float UnrealizedReturns() => 
            (((StartingValue + UnrealizedGains()) / StartingValue) - 1) * 100;

        public float RealizedReturns() =>
            (((StartingValue + RealizedGains()) / StartingValue) - 1) * 100;

        public float Returns() =>
            (((StartingValue + RealizedGains() + UnrealizedGains()) / StartingValue) - 1) * 100;


        public void Step(int index, float price, DateTime date, string side)
        {
            State.CurrentPrice = price;

            CheckRisk(index, price);

            if (side == "Buy")
            {
                Buy(index, price, date);
            }

            if (side == "Sell")
            {
                Sell(index, price, date);
            }

            if (side == "Hold")
            {
                Hold(index, price, date);
            }

            CurrentValue = Gains();
        }

        private void Hold(int index, float price, DateTime date)
        {
            Predictions.Add(new Prediction
            {
                Price = price,
                Date = date,
                Index = index,
                PortfolioCashValue = CurrentCash,
                PortfolioTotalValue = Gains(),
                Side = "Hold"
            });


        }

        private void Sell(int index, float price, DateTime date)
        {
            Predictions.Add(new Prediction
            {
                Price = price,
                Date = date,
                Index = index,
                PortfolioCashValue = CurrentCash,
                PortfolioTotalValue = Gains(),
                Side = "Sell"
            });

            for (var i = 0; i < OpenPositions.Count; i++)
            {
                var currentPosition = OpenPositions[i];

                currentPosition.CloseCash = price * currentPosition.SharesBought;
                currentPosition.CloseIndex = index;
                currentPosition.ClosePrice = price;
                currentPosition.CloseReason = "Sold";
                currentPosition.Growth = price / currentPosition.OpenPrice;

                CurrentCash += currentPosition.CloseCash;

                ClosedPositions.Add(currentPosition);
            }

            OpenPositions.Clear();
        }

        private void Buy(int index, float price, DateTime date)
        {
            if (CurrentCash - MaxCash < 0)
            {
                Broke = true;
                Hold(index, price, date);
                return;
            }

            CurrentCash -= MaxCash;

            Predictions.Add(new Prediction
            {
                Price = price,
                Date = date,
                Index = index,
                PortfolioCashValue = CurrentCash,
                PortfolioTotalValue = Gains(),
                Side = "Buy"
            });

            OpenPositions.Add(new Position
            {
                OpenPrice = price,
                OpenCash = MaxCash,
                SharesBought = MaxCash / price,
                TakeLoss = price - (price * StopLossPct),
                TakeProfit = price * TakeProfitPct,
                OpenIndex = index,
                CloseReason = "StillOpen"
            });
        }


        private void CheckRisk(int index, float price)
        {
            for (var i = 0; i < OpenPositions.Count; i++)
            {
                var currentPosition = OpenPositions[i];

                if (currentPosition.TakeLoss <= price)
                {
                    currentPosition.CloseCash = price * currentPosition.SharesBought;
                    currentPosition.CloseIndex = index;
                    currentPosition.ClosePrice = price;
                    currentPosition.CloseReason = "TakeLoss";
                    currentPosition.Growth = price / currentPosition.OpenPrice;

                    CurrentCash += currentPosition.CloseCash;

                    ClosedPositions.Add(currentPosition);
                }

                if (currentPosition.TakeProfit >= price)
                {
                    currentPosition.CloseCash = price * currentPosition.SharesBought;
                    currentPosition.CloseIndex = index;
                    currentPosition.ClosePrice = price;
                    currentPosition.CloseReason = "TakeProfit";
                    currentPosition.Growth = price / currentPosition.OpenPrice;

                    CurrentCash += currentPosition.CloseCash;

                    ClosedPositions.Add(currentPosition);
                }
            }

            OpenPositions = OpenPositions.Where(pos => pos.CloseIndex != index).ToList();
        }

    }
}