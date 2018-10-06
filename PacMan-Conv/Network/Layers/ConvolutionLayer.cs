using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using PacMan_Conv.Activations;

namespace PacMan_Conv.Network.Layers {
    public class ConvolutionLayer : Layer {
        public Matrix<double>[] Filter, Last_Output, Last_Input;
        public double[] Bias;
        public int Kernel, Channels, Features;
        public Activation Activation;

        public ConvolutionLayer(int channels, int features, int kernel, Activation activation) {
            this.Channels = channels;
            this.Features = features;
            this.Kernel = kernel;
            Filter = new Matrix<double>[Features * Channels];
            Bias = new double[Features * Channels];
            for (int i = 0; i < features * channels; i++) {
                Filter[i] = DenseMatrix.CreateRandom(kernel, kernel, new ContinuousUniform(-1, 1));
                Bias[i] = Layer.Random.NextDouble() * 2 - 1;
            }
            Activation = activation;
        }

        public override Matrix<double>[] Propagate(Matrix<double>[] input) {
            Last_Input = input;
            var results = new Matrix<double>[Features];
            for (int feat = 0; feat < Features; feat++) {
                for (int chan = 0; chan < Channels; chan++) {
                    if (results[feat] == null) results[feat] = DenseMatrix.Create(input[chan].RowCount - (Kernel-1), input[chan].ColumnCount - (Kernel - 1), 0);
                    Matrix<double> filter = Filter[feat * Channels + chan];
                    double bias = Bias[feat * Channels + chan];
                    for (int x = 0; x < input[chan].RowCount - (Kernel - 1); x++)
                        for (int y = 0; y < input[chan].ColumnCount - (Kernel - 1); y++)
                            results[feat][x, y] += (input[chan].SubMatrix(x, Kernel, y, Kernel) * filter + bias).RowSums().Sum();
                }
                results[feat].Map(Activation.Activate, results[feat]);
            }
            
            Last_Output = results;
            return results;
        }


        public override Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr) {
            var result_errors = new Matrix<double>[Channels];
            var derivativeo = new Matrix<double>[Features];
            for (int feat = 0; feat < Features; feat++) {
                var m = Last_Output[feat];
                derivativeo[feat] = new DenseMatrix(m.RowCount,m.ColumnCount);
                m.Map(Activation.DeActivate, derivativeo[feat]);

                for (int chan = 0; chan < Channels; chan++) {
                    if (result_errors[chan] == null) result_errors[chan] = DenseMatrix.Create(Last_Input[0].RowCount * Last_Input[0].ColumnCount, 1, 0);
                    Bias[feat * Channels + chan] += error[feat].ColumnSums().Sum() * lnr;
                    for (int kx = 0; kx < Kernel; kx++)
                        for (int ky = 0; ky < Kernel; ky++) {
                            double delta_kxy = 0;
                            for (int x = 0; x < Last_Input[chan].RowCount; x++)
                                for (int y = 0; y < Last_Input[chan].ColumnCount; y++) {
                                    result_errors[chan][x * Last_Input[chan].ColumnCount + y, 0] += Filter[feat * Channels + chan][Kernel - kx - 1, Kernel - ky - 1] * error[feat][ky*Kernel+kx, 0];
                                    if (y < Last_Input[chan].ColumnCount - (Kernel - 1) && x < Last_Input[chan].RowCount - (Kernel - 1))
                                        delta_kxy += Last_Input[chan][x + kx, y + ky] * error[feat][ky * Kernel + kx, 0] * lnr;
                                }
                            Filter[feat * Channels + chan][kx, ky] += delta_kxy;
                        }
                }
            }
            return result_errors;
        }
    }
}
