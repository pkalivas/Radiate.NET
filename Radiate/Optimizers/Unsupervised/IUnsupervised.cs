﻿using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Unsupervised;

public interface IUnsupervised
{
    Prediction Predict(Tensor tensor);
    Task Train(Batch batch, Func<Epoch, bool> trainFunc);
}