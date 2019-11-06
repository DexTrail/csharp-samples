using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    #region Transfer functions and their derivatives

    public enum TransferFunction
    {
        None,
        Sigmoid
    }

    static class TransferFunctions
    {
        public static double Evaluate(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid(input);

                case TransferFunction.None:
                default:
                    return 0.0;
            }
        }
        public static double EvaliateDerivative(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid_derivative(input);

                case TransferFunction.None:
                default:
                    return 0.0;
            }
        }

        /* Transfer function definitions */
        private static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        private static double sigmoid_derivative(double x)
        {
            return sigmoid(x) * (1.0 - sigmoid(x));
        }
    }

    #endregion

    public class BackPropagationNetwork
    {
        #region Constructors

        public BackPropagationNetwork(int[] layerSizes, TransferFunction[] transferFunctions)
        {
            // Validate the input data
            if (transferFunctions.Length != layerSizes.Length || transferFunctions[0] != TransferFunction.None)
                throw new ArgumentException("Cannot construct a network with these parameters.");

            // Initialize network layers
            layerCount = layerSizes.Length - 1;
            inputSize = layerSizes[0];

            // TODO: Поместить обе инициализации в один цикл
            layerSize = new int[layerCount];
            for (int i = 0; i < layerCount; i++)
                layerSize[i] = layerSizes[i + 1];

            transferFunction = new TransferFunction[layerCount];
            for (int i = 0; i < layerCount; i++)
                transferFunction[i] = transferFunctions[i + 1];

            // Start dimensioning arrays
            bias = new double[layerCount][];
            previousBiasDelta = new double[layerCount][];
            delta = new double[layerCount][];
            layerOutput = new double[layerCount][];
            layerInput = new double[layerCount][];

            weight = new double[layerCount][][];
            previousWeightDelta = new double[layerCount][][];

            // Fill 2 dimensional arrays
            for (int l = 0; l < layerCount; l++)
            {
                bias[l] = new double[layerSize[l]];
                previousBiasDelta[l] = new double[layerSize[l]];
                delta[l] = new double[layerSize[l]];
                layerOutput[l] = new double[layerSize[l]];
                layerInput[l] = new double[layerSize[l]];

                weight[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];
                previousWeightDelta[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    weight[l][i] = new double[layerSize[l]];
                    previousWeightDelta[l][i] = new double[layerSize[l]];
                }
            }

            // Initialize the weights
            for (int l = 0; l < layerCount; l++)
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
                    bias[l][j] = Gaussian.GetRandomGaussian();
                    previousBiasDelta[l][j] = 0.0;
                    layerOutput[l][j] = 0.0;
                    layerInput[l][j] = 0.0;
                    delta[l][j] = 0.0;
                }

                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weight[l][i][j] = Gaussian.GetRandomGaussian();
                        previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
        }

        #endregion

        #region Methods

        public void Run(ref double[] input, out double[] output)
        {
            // Make sure we have enough data
            if (input.Length != inputSize)
                throw new ArgumentException("Input data is not of the correct dimension");

            // Dimension
            output = new double[layerSize[layerCount - 1]];

            /* Run the network! */
            for (int l = 0; l < layerCount; l++)
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                        sum += weight[l][i][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);

                    sum += bias[l][j];
                    layerInput[l][j] = sum;

                    layerOutput[l][j] = TransferFunctions.Evaluate(transferFunction[l], layerInput[l][j]);
                }
            }

            // Copy the output to the output array
            for (int i = 0; i < layerSize[layerCount - 1]; i++)
                output[i] = layerOutput[layerCount - 1][i];
        }

        #endregion

        #region Private data

        private int layerCount; // Number of real layers (excluding the input layer)
        private int inputSize;  // Demension of the input
        private int[] layerSize;    // Dimension layerCount and holds number of nodes per layer of all the real layers
        private TransferFunction[] transferFunction;    // Same dimension as array of layers (layerSize). Specifies transfer function for the layer

        private double[][] layerOutput; // First index is the layer we concerned with and second is the node in that layer
        private double[][] layerInput;
        private double[][] bias;    // [layer][node] bias for the node
        private double[][] delta;   // Delta for node
        private double[][] previousBiasDelta;   // Delta for bias that was applied during the training iteration

        // .NET seek faster in one-dimensional arrays even though they take slightly more space. [][][] is better than [][,] for large networks
        private double[][][] weight;    // Weights. First index - layer, second - node from, third - node to
        private double[][][] previousWeightDelta;   // First index - layer, second - node from, third - node to

        #endregion

        // Have methods that return random numbers that distributed normally
        public static class Gaussian
        {
            private static Random gen = new Random();

            public static double GetRandomGaussian()
            {
                return GetRandomGaussian(0.0, 1.0);
            }

            public static double GetRandomGaussian(double mean, double stddev)
            {
                double rVal1, rVal2;

                GetRandomGaussian(mean, stddev, out rVal1, out rVal2);

                return rVal1;
            }

            // stddev - standard deviation
            public static void GetRandomGaussian(double mean, double stddev, out double val1, out double val2)
            {
                double u, v, s, t;

                do
                {
                    u = 2 * gen.NextDouble() - 1;   // [-1; 1]
                    v = 2 * gen.NextDouble() - 1;   // [-1; 1]
                } while (u * u + v * v > 1 || (u == 0 && v == 0));  // u and v randomly distributed inside the unit disc and can not be the origin (0, 0) and they can't be on the boundry at radius 1

                s = u * u + v * v;  // s - radius squared
                t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

                val1 = stddev * u * t + mean;   // (u * t) - normally distributed
                val2 = stddev * v * t + mean;   // (v * t) - normally distributed
            }
        }
    }
}
