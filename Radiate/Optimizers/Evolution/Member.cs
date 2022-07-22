
using Radiate.Optimizers.Evolution.Interfaces;

namespace Radiate.Optimizers.Evolution;

public class Member
{
    public IGenome Model { get; set; }
    public float Fitness { get; set; }
}
