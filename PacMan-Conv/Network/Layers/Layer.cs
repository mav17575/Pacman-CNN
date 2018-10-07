using System;
using MathNet.Numerics.LinearAlgebra;

namespace PacMan_Conv.Network.Layers {
    [Serializable]
    //structure for all layers
    abstract public class Layer {
        public static Random Random = new Random();

        abstract public Matrix<double>[] Propagate(Matrix<double>[] input);

        abstract public Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr);
    }
}
