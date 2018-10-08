using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PacMan_Conv.Activations;
using PacMan_Conv.LossFunctions;
using PacMan_Conv.NeuralNetwork;
using PacMan_Conv.NeuralNetwork.Layers;

namespace PacMan_Conv {
    class Program {
        /// <summary>
        /// only for test purpose
        /// </summary>
        /// <forDevs>
        /// code to test only in here
        /// </forDevs>
        /// <param name="args">The command-line arguments.</param>
        static void Main(string[] args) {
            MathNet.Numerics.Control.UseMultiThreading();
           // MathNet.Numerics.Control.TryUseNativeMKL();

            var input = ImgUtil.GetPixelsRGB(new Bitmap("../../../../cnnimg.jpg"));//DenseMatrix.CreateRandom(400, 400, RandomUtil.UnitUniform);
            //var input = new Matrix<double>[] { m };

            var c1 = new ConvolutionLayer(3, 6, 3, new Sigmoid());
            var m1 = new MaxPoolingLayer(3);
            var c2 = new ConvolutionLayer(6, 12, 2, new Sigmoid());
            var m2 = new MaxPoolingLayer(3);
            var c3 = new ConvolutionLayer(12, 12, 5, new Sigmoid());
            var d1 = new DenseLayer(38*71*12, 1000, new Sigmoid(), true);
            var d2 = new DenseLayer(1000, 100, new Sigmoid(), false);
            var d3 = new DenseLayer(100, 1, new Sigmoid(), false);

            var network = new Network(new MeanSquaredError(), new List<Layer> {c1, m1, c2, m2, c3, d1, d2, d3});

            /*Console.WriteLine(network.Propagate(input)[0]);
            Network.SaveNetwork(network);
            network = Network.LoadNetwork();
            Console.WriteLine(network.Propagate(input)[0]);
            */

            Stopwatch watch = new Stopwatch();
            watch.Start();

            for (int a = 0; a < 60; a++) {
                Stopwatch wat = new Stopwatch();
                wat.Start();
                var result = network.Propagate(input);
                wat.Stop();
                Console.WriteLine("-------------#" + a + "--------------");
                Console.WriteLine("Time Propagate: " + wat.ElapsedMilliseconds + "ms");
                Console.WriteLine("");
                Console.WriteLine(result[0]);
                Console.WriteLine("");
                wat.Restart();
                wat.Start();
                var backward_result = network.Backpropagate(network.Target.Target(new Matrix<double>[] { DenseMatrix.Create(1, 1, 1) }, result), 0.001);
                wat.Stop();
                Console.WriteLine("Time Backpropagate: " + wat.ElapsedMilliseconds + "ms");
                Console.WriteLine("-----------------------------");
                ImgUtil.ConvertToBitmap(c1.Last_Output[2], c1.Last_Output[2].ColumnCount, c1.Last_Output[2].RowCount,"t"+a,"Sigmoid");
            }
            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);
            Console.ReadLine();
        }
        
    }
}
