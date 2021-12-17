namespace Radiate.Data.Charting;

public class ChartRequest
{
    public string Title { get; set; }
    public string YOneName { get; set; }
    public string YTwoName { get; set; }
    public List<float> YOneData { get; set; }
    public List<float> YTwoData { get; set; }
}