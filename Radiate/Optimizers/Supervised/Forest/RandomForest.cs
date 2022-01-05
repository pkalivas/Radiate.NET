using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Forest;

public class RandomForest : ISupervised
{
    private readonly int _nTrees;
    private readonly ForestInfo _info;
    private readonly DecisionTree[] _trees;
    
    public RandomForest(int nTrees, ForestInfo info)
    {
        _nTrees = nTrees;
        _info = info;
        _trees = new DecisionTree[nTrees];
    }

    public Task Train(List<Batch> data, LossFunction lossFunction, Func<Epoch, bool> trainFunc)
    {
        var batch = data.First();
        for (var i = 0; i < _nTrees; i++)
        {
            _trees[i] = new DecisionTree(_info, batch);
        }
        
        throw new NotImplementedException();
    }
    
    public Prediction Predict(Tensor input)
    {
        throw new NotImplementedException();
    }
}