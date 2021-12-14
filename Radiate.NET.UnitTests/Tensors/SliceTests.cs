﻿using FluentAssertions;
using Radiate.NET.Domain.Tensors;
using Radiate.NET.UnitTests.Utils;
using Xunit;

namespace Radiate.NET.UnitTests.Tensors
{
    public class SliceTests
    {
        [Fact]
        public void Slice_Depth_Test()
        {
            var testTensor = LayerUtils.EightEightOneTensor;

            var hDiff = new[] { 2, 4 };
            var wDiff = new[] { 2, 4 };
            var dDiff = new[] { 0, testTensor.Shape.Depth };
            var tensorSlice = testTensor.Slice(hDiff, wDiff, dDiff);

            tensorSlice.Shape.Depth.Should().Be(testTensor.Shape.Depth);
        }
    }
}