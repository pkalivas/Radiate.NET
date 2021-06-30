namespace Radiate.NET.Environments.MarketEnvironment.Strategies
{
    public interface IStrategy
    {
        float GetReward(Portfolio portfolio);
    }
}