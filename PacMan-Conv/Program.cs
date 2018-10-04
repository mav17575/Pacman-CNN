using System;
using System.Diagnostics;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PacMan_Conv.Layers;

namespace PacMan_Conv
{
    class Program
    {
        static void Main(string[] args)
        {
            MathNet.Numerics.Control.UseMultiThreading();


            /*var input = new Matrix<double>[] { DenseMatrix.Create(4, 4, 1), DenseMatrix.Create(4, 4, 1), DenseMatrix.Create(4, 4, 1) };

            var convlayer = new Layers.ConvolutionLayer(3, 2, 2);
            var convlayer2 = new Layers.ConvolutionLayer(2, 1, 2);

            var res = convlayer.propagate(input);
            res = convlayer2.propagate(res);

            Console.WriteLine(res[0]);

            for (int a = 0; a < 100; a++)
            {
                var error = new Matrix<double>[] { DenseMatrix.Create(4, 1, (DenseMatrix.Create(2, 2, 1) - res[0])[0, 0]) };

                error = convlayer2.backward(error, 0.0001);
                error = convlayer.backward(error, 0.0001);

                res = convlayer.propagate(input);
                res = convlayer2.propagate(res);

                Console.WriteLine(res[0]);
            }*/


            var input_wert = GetPixelsGray(new Bitmap("/Users/timoluick/Projects/PacMan-Conv/PacMan-Conv/dog.jpg"));

            var input = new Matrix<double>[] { input_wert, input_wert, input_wert };

            var convLayer1 = new ConvolutionLayer(3, 6, 5);
            var convLayer2 = new ConvolutionLayer(6, 12, 3);
            var convLayer3 = new ConvolutionLayer(12, 16, 3);

            Stopwatch watch = new Stopwatch();
            watch.Start();
            input = convLayer1.propagate(input);
            input = convLayer2.propagate(input);
            input = convLayer3.propagate(input);
            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);

            /*Stopwatch watch = new Stopwatch();
            watch.Start();
            Console.WriteLine(input[0]);
            for (int a = 0; a < 10; a++)
            {
                var result = convlayer.propagate(input);
                var error = new Matrix<double>[] { DenseMatrix.Create(4, 1, (DenseMatrix.Create(2, 2, 1) - result[0])[0, 0]), DenseMatrix.Create(4, 1, (DenseMatrix.Create(2, 2, 1) - result[1])[0, 0]) };
                convlayer.backward(error, 0.01);
                Console.WriteLine("Result 1");
                Console.WriteLine(result[0]);
                Console.WriteLine("Result 2");
                Console.WriteLine(result[1]);
            }
            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);*/
        }

        public static Matrix<double> GetPixelsGray(Bitmap img)
        {
            Matrix<double> res = new DenseMatrix(img.Height, img.Width);
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    Color c = img.GetPixel(x, y);
                    double g = (c.R + c.G + c.B) / (765.0);
                    g = Math.Round(g, 3);
                    res[y, x] = g;
                }
            }
            return res;
        }
    }
}
