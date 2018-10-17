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
                Filter[i] = DenseMatrix.CreateRandom(1, kernel * kernel, RandomUtil.UnitUniform);
                Bias[i] = RandomUtil.NormalRandom.NextDouble() * 2 - 1;
            }
            Activation = activation;
        }

        public override Matrix<double>[] Propagate(Matrix<double>[] input)
        {
            Lastkx = input[0].ColumnCount - Kernel + 1;
            Lastky = input[0].RowCount - Kernel + 1;
            //Stopwatch watch = new Stopwatch();
            //watch.Start();
            Last_Input = input;
            Matrix<double>[] conv = ApplyKernel(input);
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
            //watch.Stop();
            //Console.WriteLine("ConvolutionLayer; Calculation:" + watch.ElapsedMilliseconds + "ms");
            //watch.Reset();
            Last_Output = results;
            return results;
        }


        public override Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr)  {
            var result_errors = new Matrix<double>[Channels];
            var derivative = new Matrix<double>[Features];
            for (int feat = 0; feat < Features; feat++) {
                derivative[feat] = Last_Output[feat].Map(Activation.Derivative);
                Matrix<double> g = error[feat].PointwiseMultiply(new DenseMatrix(derivative[feat].RowCount * derivative[feat].ColumnCount, 1, derivative[feat].AsColumnMajorArray()));

                for (int chan = 0; chan < Channels; chan++) {
                    if (result_errors[chan] == null) result_errors[chan] = DenseMatrix.Create(Last_Input[0].RowCount * Last_Input[0].ColumnCount, 1, 0);
                    Bias[feat * Channels + chan] += g.ColumnSums().Sum() * lnr;
                    for (int kx = 0; kx < Kernel; kx++) {
                        for (int ky = 0; ky < Kernel; ky++) {
                            int errorIndex = 0;
                            double delta_kxy = 0;
                            for (int x = 0; x < Last_Input[chan].RowCount; x++)
                                for (int y = 0; y < Last_Input[chan].ColumnCount; y++) {
                                    if (x - kx >= 0 && y - ky >= 0 && x + (Kernel - kx) < Last_Input[chan].RowCount && y + (Kernel - ky) < Last_Input[chan].ColumnCount) {
                                        result_errors[chan][x * Last_Input[chan].ColumnCount + y, 0] += Filter[feat * Channels + chan][0, (Kernel - kx - 1) * Kernel + (Kernel - ky - 1)] * g[errorIndex, 0];
                                        errorIndex++;
                                    }
                                    if (y < Last_Input[chan].ColumnCount - (Kernel - 1) && x < Last_Input[chan].RowCount - (Kernel - 1))
                                        delta_kxy += Last_Input[chan][x + kx, y + ky] * g[kx * Kernel + ky, 0] * lnr;
                                }
                            Filter[feat * Channels + chan][0, kx * Kernel + ky] += delta_kxy;
                        }
                    }
                }
            }
            return result_errors;
        }

        public Matrix<double>[] ApplyKernel(Matrix<double>[] input) {
            var result = new Matrix<double>[Channels];

            int countx = input[0].ColumnCount - Kernel + 1;
            int county = input[0].RowCount - Kernel + 1;

            int kxm = Kernel - 1;
            int r = Kernel * Kernel;
            int c = countx * county;

            for (int chan = 0; chan < Channels; chan++) {
                result[chan] = new DenseMatrix(r, c);
                for (int y = 0; y < county; y++) {
                    for (int x = 0; x < countx; x++) {
                        int offx = 0;
                        int i = x + (y * countx);
                        for (int ky = 0; ky < Kernel; ky++) {
                            int kfy = ky + offx;
                            int yky = y + ky;
                            for (int kx = 0; kx < Kernel; kx++)
                                result[chan][kfy + kx, i] += input[chan][yky, x + kx];
                            offx += kxm;
                        }
                    }
                }
            }
            return result;
        }
    }
}
