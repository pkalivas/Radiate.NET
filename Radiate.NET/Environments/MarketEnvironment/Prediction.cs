using System;

namespace Radiate.NET.Environments.MarketEnvironment
{
    public class Prediction
    {
        public float Price { get; set; }
        public string Side { get; set; }
        public DateTime Date { get; set; }
        public int Index { get; set; }
        public float PortfolioTotalValue { get; set; }
        public float PortfolioCashValue { get; set; }
    }
}