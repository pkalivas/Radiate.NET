using Radiate.Callbacks.Interfaces;

namespace Radiate.Callbacks;

public static class CallbackResolver
{
    public static List<T> Get<T>(IEnumerable<ITrainingCallback> callbacks)
    {
        var result = new List<T>();

        foreach (var callback in callbacks)
        {
            if (callback is T back)
            {
                result.Add(back);
            }
        }

        return result;
    }
    
}