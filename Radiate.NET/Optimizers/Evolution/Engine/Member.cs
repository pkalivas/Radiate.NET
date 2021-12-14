
namespace Radiate.NET.Optimizers.Evolution.Engine
{
    public class Member<T>
    {
        public T Model { get; set; }
        public double Fitness { get; set; }
    }
}