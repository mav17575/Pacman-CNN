using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using PacMan_Conv.Activations;

namespace PacMan_Conv.Layers {
    public class DenseLayer : Layer {
        Matrix<double> Weight, Bias, Last_Output, Last_Input;
        readonly Activation Activation;

        public DenseLayer(int input_size, int output_size, Activation activation_function) {
            Weight = DenseMatrix.CreateRandom(output_size, input_size, new ContinuousUniform(-1, 1));
            Bias = DenseMatrix.CreateRandom(output_size, 1, new ContinuousUniform(-1, 1));
            Activation = activation_function;
        }

        public override Matrix<double>[] propagate(Matrix<double>[] input) {
            Last_Input = input[0];
            input[0] = input[0] * Weight + Bias;
            input[0] = input[0].Map(Activation.Activate);
            Last_Output = input[0];
            return new Matrix<double>[] { input[0] };
        }

        public override Matrix<double>[] backward(Matrix<double>[] error, double lnr) {
            var gradient = error[0].PointwiseMultiply(Last_Output.Map(Activation.DeActivate)) * lnr;

            error[0] = Weight.Transpose() * error[0];

            Bias += gradient;
            Weight += gradient * Last_Input.Transpose();

            return error;
        }
    }
}
