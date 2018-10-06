﻿using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using PacMan_Conv.Activations;

namespace PacMan_Conv.Network.Layers {
    public class DenseLayer : Layer {
        Matrix<double> Weight, Bias, Last_Output, Last_Input;
        readonly Activation Activation;
        readonly bool IsInputConvolution;
        int Channels, SizeRows, SizeColumns;

        public DenseLayer(int input_size, int output_size, Activation activation_function, bool isInputConvoltion) {
            Weight = DenseMatrix.CreateRandom(output_size, input_size, new ContinuousUniform(-1, 1));
            Bias = DenseMatrix.CreateRandom(output_size, 1, new ContinuousUniform(-1, 1));
            Activation = activation_function;
            this.IsInputConvolution = isInputConvoltion;
        }

        public override Matrix<double>[] Propagate(Matrix<double>[] input) {
            if (IsInputConvolution)
                input = ToVectorInput(input);
            Last_Input = input[0];
            input[0] =  Weight * input[0] + Bias;
            input[0] = input[0].Map(Activation.Activate);
            Last_Output = input[0];
            return new Matrix<double>[] { input[0] };
        }

        public override Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr) {
            var gradient = error[0].PointwiseMultiply(Last_Output.Map(Activation.DeActivate)) * lnr;

            error[0] = Weight.Transpose() * error[0];

            Bias += gradient;
            Weight += gradient * Last_Input.Transpose();

            if (IsInputConvolution)
                error = ToMatrixError(error);
            return error;
        }

        Matrix<double>[] ToVectorInput(Matrix<double>[] input) {
            Channels = input.Length;
            Matrix<double>[] result = new Matrix<double>[1];
            SizeRows = input[0].RowCount;
            SizeColumns = input[0].ColumnCount;
            int off = SizeRows * SizeColumns;
            result[0] = DenseMatrix.Create(off * input.Length, 1, 0);
            for (int i = 0; i < input.Length; i++) {
                var x = new DenseMatrix(off, 1, input[i].AsColumnMajorArray());
                result[0].SetSubMatrix(off * i, off, 0, 1, x);
            }
            return result;
        }

        Matrix<double>[] ToMatrixError(Matrix<double>[] error){
            var result = new Matrix<double>[Channels];
            for (int chan = 0; chan < Channels; chan++)
                result[chan] = error[0].SubMatrix(chan * SizeRows * SizeColumns, SizeRows * SizeColumns, 0, 1);
            return result;
        } 
    }
}