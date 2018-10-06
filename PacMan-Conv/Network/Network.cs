using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using PacMan_Conv.Network.Layers;

namespace PacMan_Conv.Network {
    /// <summary>
    /// network can also be used as a layer in a network
    /// </summary>
    public class Network : Layer {
        /// <summary>
        /// all layers of a network
        /// </summary>
        List<Layer> Layers;

        public Network(List<Layer> layers) {
            this.Layers = layers;
        }

        public override Matrix<double>[] Propagate(Matrix<double>[] input) {
            foreach(var layer in Layers) {
                input = layer.Propagate(input);
            }
            return input;
        }

        public override Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr) {
            Layers.Reverse();
            foreach (var layer in Layers)
                error = layer.Backpropagate(error, lnr);
            Layers.Reverse();
            return error;
        }
    }
}
