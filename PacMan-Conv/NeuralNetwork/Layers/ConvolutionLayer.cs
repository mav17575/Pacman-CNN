using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using PacMan_Conv.Activations;
using System;
using System.Diagnostics;

namespace PacMan_Conv.NeuralNetwork.Layers
{
    [Serializable]
    public class ConvolutionLayer : Layer
    {
        /// <summary>
        /// Filter: all weight matrices are in there
        /// LastOutput: last calculated output -> for backprop
        /// LastInput: last input to calculate -> for backprop
        /// </summary>
        public Matrix<double>[] Filter, Last_Output, Last_Input;
        public double[] Bias;
        /// <summary>
        /// Kernel: size of the kernel
        /// Channels: number of inputs accepted
        /// Features: number of outputs
        /// </summary>
        public int Kernel, Channels, Features;
        /// <summary>
        /// Activation -> example: sigmoid, leakyReLu
        /// </summary>
        public int Lastkx,Lastky;
        public Activation Activation;

        public ConvolutionLayer(int channels, int features, int kernel, Activation activation)
        {
            this.Channels = channels;
            this.Features = features;
            this.Kernel = kernel;
            Filter = new Matrix<double>[Features * Channels];
            Bias = new double[Features * Channels];
            for (int i = 0; i < features * channels; i++)
            {
                //Filter[i] = DenseMatrix.CreateRandom(kernel, kernel, RandomUtil.UnitUniform);
                Filter[i] = DenseMatrix.CreateRandom(1, kernel * kernel, RandomUtil.UnitUniform);
                Bias[i] = RandomUtil.NormalRandom.NextDouble() * 2 - 1;
            }
            Activation = activation;
        }

        public override Matrix<double>[] Propagate(Matrix<double>[] input)
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            Last_Input = input;
            Matrix<double>[] conv = IConvolute(input);
            var results = new Matrix<double>[Features];
            for (int feat = 0; feat < Features; feat++)
            {
                results[feat] = new DenseMatrix(1,Lastkx*Lastky);
                for (int chan = 0; chan < Channels; chan++)
                {
                    results[feat] += Filter[feat * Channels+chan] * conv[chan];
                }
                results[feat].Map(Activation.Activate, results[feat]);
                results[feat] = new DenseMatrix(Lastkx,Lastky,results[feat].AsColumnMajorArray()).Transpose();
            }
            watch.Stop();
            Console.WriteLine("ConvolutionLayer; Calculation:" + watch.ElapsedMilliseconds + "ms");
            watch.Reset();
            Last_Output = results;
            return results;
        }


        public override Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr)
        {
            var result_errors = new Matrix<double>[Channels];
            var derivativeo = new Matrix<double>[Features];
            for (int feat = 0; feat < Features; feat++)
            {
                var m = Last_Output[feat];
                derivativeo[feat] = m.Map(Activation.Derivative);

                for (int chan = 0; chan < Channels; chan++)
                {
                    if (result_errors[chan] == null) result_errors[chan] = DenseMatrix.Create(Last_Input[0].RowCount * Last_Input[0].ColumnCount, 1, 0);
                   // Bias[feat * Channels + chan] += error[feat].ColumnSums().Sum() * lnr;
                    for (int kx = 0; kx < Kernel; kx++)
                        for (int ky = 0; ky < Kernel; ky++)
                        {
                            double delta_kxy = 0;
                            for (int x = 0; x < Last_Input[chan].RowCount; x++)
                                for (int y = 0; y < Last_Input[chan].ColumnCount; y++)
                                {
                                    //dero koennte falsch sein
                                    //Sachen die wiederverwendet werden in variablen speichern
                                    double dero = derivativeo[feat][(ky+y)/Kernel, (kx + x) / Kernel];
                                    Bias[feat * Channels + chan] += error[feat][ky * Kernel + kx, 0]*dero;
                                    result_errors[chan][x * Last_Input[chan].ColumnCount + y, 0] += Filter[feat * Channels + chan][0,(Kernel - kx - 1)*Kernel+(Kernel - ky - 1)] * error[feat][ky * Kernel + kx, 0]*dero;
                                    if (y < Last_Input[chan].ColumnCount - (Kernel - 1) && x < Last_Input[chan].RowCount - (Kernel - 1))
                                        delta_kxy += Last_Input[chan][x + kx, y + ky] * error[feat][ky * Kernel + kx, 0] * dero*lnr;
                                }
                            Filter[feat * Channels + chan][0,kx*Kernel + ky] += delta_kxy;
                        }
                }
            }
            return result_errors;
        }

        public Matrix<double>[] IConvolute(Matrix<double>[] m)
        {
            Lastkx = (m[0].ColumnCount - Kernel + 1);
            Lastky = (m[0].RowCount - Kernel + 1);
            Matrix[] conv = new Matrix[Channels];
            ApplyKernel(conv, m, Lastkx, Lastky, Kernel, Kernel, Channels);
            return conv;
        }

        public static void ApplyKernel(Matrix<double>[] conv, Matrix<double>[] inputs, int countx, int county, int kernelx, int kernely, int channels)
        {
            int kxm = kernelx - 1;
            int r = kernelx * kernely;
            int c = countx * county;

            for (int z = 0; z < channels; z++)
            {
                conv[z] = new DenseMatrix(r, c);
                for (int y = 0; y < county; y++)
                {
                    for (int x = 0; x < countx; x++)
                    {
                        int offx = 0;
                        int i = x + (y * countx);
                        for (int ky = 0; ky < kernely; ky++)
                        {
                            int kfy = ky + offx;
                            int yky = y + ky;
                            for (int kx = 0; kx < kernelx; kx++)
                            {
                                conv[z][kfy + kx, i] += inputs[z].At(yky, x + kx);
                            }
                            offx += kxm;
                        }
                    }
                }
            }
        }
    }
}
