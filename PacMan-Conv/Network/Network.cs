using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using PacMan_Conv.Network.Layers;

namespace PacMan_Conv.Network {
    public class Network {
        List<Layer> Layers;

        public Network(List<Layer> layers) {
            this.Layers = layers;
        }

        public Matrix<double>[] Propagate(Matrix<double>[] input) {
            foreach(var layer in Layers) {
                input = layer.Propagate(input);
            }
            return input;
        }
    }
}
