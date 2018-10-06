using System;
using MathNet.Numerics.LinearAlgebra;

namespace PacMan_Conv.LossFunctions {
    public class MeanSquaredError : LossFunction {

        public override Matrix<double>[] Target(params Matrix<double>[][] inputs) {
            var err = inputs[0][0] - inputs[1][0];
            return new Matrix<double>[] { err };
        }
    }
}
