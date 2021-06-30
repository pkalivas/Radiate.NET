namespace Radiate.NET.Environments.MarketEnvironment
{
    public class Position
    {
        public float OpenPrice { get; set; }
        public float ClosePrice { get; set; }
        public float OpenCash { get; set; }
        public float CloseCash { get; set; }
        public float SharesBought { get; set; }
        public float TakeLoss { get; set; }
        public float TakeProfit { get; set; }
        public float Growth { get; set; }
        public int OpenIndex { get; set; }
        public int CloseIndex { get; set; }
        public string CloseReason { get; set; }
    }
}