using System.Collections.Generic;

namespace Radiate.Optimizers.Evolution.Neat
{
    public enum UpdateType
    {
        Pending,
        Activated
    }

    public class NodeUpdate
    {
        public UpdateType UpdateType { get; set; }
        public Pending Pending { get; set; }
        public Activated Activated { get; set; }

        public bool IsPending() => UpdateType == UpdateType.Pending;

        public int Output() => Pending.Output;

        public (float value, int output) IsActivated() => UpdateType switch
        {
            UpdateType.Activated => (Activated.Value, Activated.Output),
            UpdateType.Pending => (0, 0)
        };

        public static NodeUpdate Process(List<NodeUpdate> updates, Neuron node, int output)
        {
            var sum = node.Bias;
            var pendingInputs = 0;

            foreach (var edge in node.IncomingEdges())
            {
                if (updates.Count > edge.Src.Index)
                {
                    if (updates[edge.Src.Index].UpdateType == UpdateType.Activated)
                    {
                        sum += updates[edge.Src.Index].Activated.Value * edge.Weight;
                    }
                    else
                    {
                        pendingInputs++;
                    }
                }
                else
                {
                    pendingInputs++;
                }
            }

            if (pendingInputs == 0)
            {
                node.CurrentState = sum;
                node.Activate();
                
                return new NodeUpdate
                {
                    UpdateType = UpdateType.Activated,
                    Activated = new Activated
                    {
                        Value = node.ActivatedValue,
                        Output = output
                    }
                };
            }

            return new NodeUpdate
            {
                UpdateType = UpdateType.Pending,
                Pending = new Pending
                {
                    Sum = sum,
                    PendingInputs = pendingInputs,
                    Output = output
                }
            };
        }
    }
}