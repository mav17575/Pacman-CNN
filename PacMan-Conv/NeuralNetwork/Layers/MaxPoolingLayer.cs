using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Linq;

namespace PacMan_Conv.NeuralNetwork.Layers {
    [Serializable]
    public class MaxPoolingLayer : Layer {
        readonly int Kernel;
        Matrix<double>[] LastInput, LastOutput;
        int InputColumnSize, InputRowSize;

        public MaxPoolingLayer(int kernel) {
            this.Kernel = kernel;
        }

        public override Matrix<double>[] Propagate(Matrix<double>[] input) {
            LastInput = new Matrix<double>[input.Length];
            InputRowSize = input[0].RowCount;
            InputColumnSize = input[0].ColumnCount;

            var result = new Matrix<double>[input.Length];

            int SizeRows = input[0].RowCount / Kernel;
            if ((input[0].RowCount / (double)Kernel) > SizeRows)
                SizeRows++;
            int SizeColumn = input[0].ColumnCount / Kernel;
            if ((input[0].ColumnCount / (double)Kernel) > SizeColumn)
                SizeColumn++;

            for (int chan = 0; chan < input.Length; chan++) {
                result[chan] = DenseMatrix.Create(SizeRows, SizeColumn, 0);
                var input_matrix = DenseMatrix.Create(SizeRows * Kernel, SizeColumn * Kernel, 0);
                input_matrix.SetSubMatrix(0, 0, input[chan].SubMatrix(0, input[chan].RowCount, 0, input[chan].ColumnCount));
                for (int x = 0; x < SizeRows*Kernel; x += Kernel)
                    for (int y = 0; y < SizeColumn*Kernel; y += Kernel)
                        result[chan][x/Kernel, y/Kernel] = input_matrix.SubMatrix(x, Kernel, y, Kernel).AsColumnMajorArray().Max();
                LastInput[chan] = input_matrix;
            }
            LastOutput = result;
            return result;
        }

        public override Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr) {
            var newError = new Matrix<double>[LastInput.Length];

            for (int chan = 0; chan < LastInput.Length; chan++) {
                newError[chan] = DenseMatrix.Create(InputRowSize * InputColumnSize, 1, 0);
                for (int x = 0; x < LastOutput[chan].RowCount; x++) {
                    for (int y = 0; y < LastOutput[chan].ColumnCount; y++) {
                        var submatrix = LastInput[chan].SubMatrix(x * Kernel, Kernel, y * Kernel, Kernel);
                        var ma = submatrix.ToRowMajorArray();
                        int pos = Array.IndexOf(ma, LastOutput[chan][x, y]) + (x * InputColumnSize + y * Kernel);
                        newError[chan][pos, 0] = error[chan][x * LastOutput[chan].ColumnCount + y, 0];
                    }
                }
            }

            return newError;
        }
    }
}
