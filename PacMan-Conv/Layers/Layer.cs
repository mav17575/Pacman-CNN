using System;
using MathNet.Numerics.LinearAlgebra;

namespace PacMan_Conv.Layers {
    abstract public class Layer {
        public static Random Random = new Random();

        abstract public Matrix<double>[] propagate(Matrix<double>[] input);

        abstract public Matrix<double>[] backward(Matrix<double>[] error, double lnr);
    }
}
