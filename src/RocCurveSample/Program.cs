using System;
using Accord;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Analysis;
namespace RocCurveSample
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] inputs =
            {
                // Those are from class -1
                new double[] { 2, 4, 0 },
                new double[] { 5, 5, 1 },
                new double[] { 4, 5, 0 },
                new double[] { 2, 5, 5 },
                new double[] { 4, 5, 1 },
                new double[] { 4, 5, 0 },
                new double[] { 6, 2, 0 },
                new double[] { 4, 1, 0 },

                // Those are from class +1
                new double[] { 1, 4, 5 },
                new double[] { 7, 5, 1 },
                new double[] { 2, 6, 0 },
                new double[] { 7, 4, 7 },
                new double[] { 4, 5, 0 },
                new double[] { 6, 2, 9 },
                new double[] { 4, 1, 6 },
                new double[] { 7, 2, 9 },
            };

            int[] outputs =
            {
                -1, -1, -1, -1, -1, -1, -1, -1, // fist eight from class -1
                +1, +1, +1, +1, +1, +1, +1, +1  // last eight from class +1
            };

            // Next, we create a linear Support Vector Machine with 4 inputs
            SupportVectorMachine machine = new SupportVectorMachine(inputs: 3);

            // Create the sequential minimal optimization learning algorithm
            var smo = new SequentialMinimalOptimization(machine, inputs, outputs);

            // We learn the machine
            double error = smo.Run();

            // And then extract its predicted labels
            double[] predicted = new double[inputs.Length];
            for (int i = 0; i < predicted.Length; i++)
                predicted[i] = machine.Compute(inputs[i]);

            // At this point, the output vector contains the labels which
            // should have been assigned by the machine, and the predicted
            // vector contains the labels which have been actually assigned.

            // Create a new ROC curve to assess the performance of the model
            var roc = new ReceiverOperatingCharacteristic(outputs, predicted);
            roc.Compute(100); // Compute a ROC curve with 100 cut-off points
            roc.GetScatterplot(true);
            Console.WriteLine(roc.Area.ToString());
            Console.Write(roc.StandardError.ToString());
        }
    }
}
