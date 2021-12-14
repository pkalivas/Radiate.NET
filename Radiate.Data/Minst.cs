using System.Collections.Generic;

namespace Radiate.Data
{
    public class Minst : IDataSet
    {
        private readonly int _featureLimit;

        public Minst(int featureLimit)
        {
            
        }
        
        public (List<float[]> inputs, List<float[]> targets) GetDataSet()
        {
            throw new System.NotImplementedException();
        }
    }
}