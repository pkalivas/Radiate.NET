
namespace Radiate.Optimizers.Evolution.Engine.Delegates
{
    public delegate double Solve<in T>(T model) where T : Genome;

}