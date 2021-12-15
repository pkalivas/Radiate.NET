using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.UnitTests.Tensors;

public class TensorTests
{
    [Fact]
    public void Tensor_Max_is_Max()
    {
        var threeD = new Shape(5, 5, 5);
        var three = Tensor.Fill(threeD, 0f);

        three[0, 0, 3] = 5f;
        var threeMax = three.Max();

        var twoD = new Shape(5, 5);
        var two = Tensor.Fill(twoD, 0);
        two[0, 2] = 5;
        two[1, 1] = 4;
        two[1, 0] = -10;
        var twoMax = two.Max();

        var oneD = new Shape(5);
        var one = Tensor.Fill(oneD, 0);
        one[0] = 5;
        one[1] = -5;
        var oneMax = one.Max();
        
        threeMax.Should().Be(5f);
        twoMax.Should().Be(5);
        oneMax.Should().Be(5);
    }

    [Fact]
    public void Tensor_Summatory_Test()
    {
        
    }
}