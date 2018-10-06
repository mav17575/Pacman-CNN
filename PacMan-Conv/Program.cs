using System;
using System.Diagnostics;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PacMan_Conv.Network.Layers;

namespace PacMan_Conv
{
    class Program
    {
        static void Main(string[] args)
        {
            MathNet.Numerics.Control.UseMultiThreading();

            var m = DenseMatrix.Create(2, 2, 1);
            m[1, 1] = 100;
            var input = new Matrix<double>[] { m };

            MaxPoolingLayer layer = new MaxPoolingLayer(2);

            Stopwatch watch = new Stopwatch();
            watch.Start();
            var result = layer.Propagate(input);
            var backward_result = layer.Backpropagate(new Matrix<double>[] { DenseMatrix.Create(1, 1, -0.5) }, 0.001);
            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);
            Console.WriteLine(result[0]);
            Console.WriteLine(backward_result[0]);

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
