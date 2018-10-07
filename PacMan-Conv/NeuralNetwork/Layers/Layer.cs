using System;
using MathNet.Numerics.LinearAlgebra;

namespace PacMan_Conv.NeuralNetwork.Layers {
    [Serializable]
    //structure for all layers
    abstract public class Layer {

        abstract public Matrix<double>[] Propagate(Matrix<double>[] input);

        abstract public Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr);
    }
}
