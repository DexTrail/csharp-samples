using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace NeuralNetworkTutuorialApp
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] layerSizes = new int[3] { 1, 2, 1 };
            TransferFunction[] tFunc = new TransferFunction[3]{TransferFunction.None,
                                                               TransferFunction.Sigmoid,
                                                               TransferFunction.Sigmoid};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, tFunc);

            Console.ReadLine();
        }
    }
}
