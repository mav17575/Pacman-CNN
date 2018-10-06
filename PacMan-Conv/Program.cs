using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PacMan_Conv.Activations;
using PacMan_Conv.Network.Layers;
using PacMan_Conv.Network;

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

            var m = DenseMatrix.Create(400, 400, 1);
            var input = new Matrix<double>[] { m, m, m };

            var l1 = new ConvolutionLayer(3, 6, 3, new Sigmoid());
            var l2 = new MaxPoolingLayer(2);
            var l3 = new ConvolutionLayer(6, 12, 2, new Sigmoid());
            var l4 = new MaxPoolingLayer(2);
            var l5 = new ConvolutionLayer(12, 1, 5, new Sigmoid());
            var l6 = new DenseLayer(9025, 1000, new Sigmoid(), true);
            var l7 = new DenseLayer(1000, 10, new Sigmoid(), false);
            var l8 = new DenseLayer(10, 1, new Sigmoid(), false);

            var network = new Network.Network(new List<Layer> {l1, l2, l3, l4, l5, l6, l7, l8});

            Stopwatch watch = new Stopwatch();
            watch.Start();

            for (int a = 0; a < 10000; a++) {
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
                var backward_result = network.Backpropagate(new Matrix<double>[] { DenseMatrix.Create(1, 1, 1) - result[0] }, 0.1);
                wat.Stop();
                Console.WriteLine("Time Backpropagate: " + wat.ElapsedMilliseconds + "ms");
                Console.WriteLine("-----------------------------");
            }
            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);
        }

        /// <summary>
        /// will get moved in a own class (ImageUtils)
        /// </summary>
        /// <returns>matrix with gray values between 0 and 1</returns>
        /// <param name="img">Image as Bitmap</param>
        public static Matrix<double> GetPixelsGray(Bitmap img) {
            Matrix<double> result = new DenseMatrix(img.Height, img.Width);
            for (int y = 0; y < img.Height; y++)
                for (int x = 0; x < img.Width; x++) {
                    Color c = img.GetPixel(x, y);
                    result[y, x] = (c.R + c.G + c.B) / (765.0);
                }
            return result;
        }
    }
}
