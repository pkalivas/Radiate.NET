using System.Linq;

namespace Radiate.NET.Environments.MarketEnvironment.Strategies
{
    public class TotalReturnStrategy : IStrategy
    {
        public float GetReward(Portfolio portfolio)
        {
            var firstValue = portfolio.Predictions.First().PortfolioTotalValue;
            var lastValue = portfolio.Predictions.Last().PortfolioTotalValue;

            return ((lastValue / firstValue) - 1) * 100;
        }
    }
}