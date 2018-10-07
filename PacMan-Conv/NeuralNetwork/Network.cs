using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using MathNet.Numerics.LinearAlgebra;
using PacMan_Conv.LossFunctions;
using PacMan_Conv.NeuralNetwork.Layers;

namespace PacMan_Conv.NeuralNetwork {
    [Serializable]
    /// <summary>
    /// network can also be used as a layer in a network
    /// </summary>
    public class Network : Layer {
        /// <summary>
        /// all layers of a network
        /// </summary>
        public List<Layer> Layers;
        public LossFunction Target;

        public Network(LossFunction target, params Layer[] layers)
        {
            this.Layers = new List<Layer>(layers);
            Target = target;
        }
        public Network(LossFunction target, List<Layer> layers) {
            this.Layers = layers;
            Target = target;
        }

        public override Matrix<double>[] Propagate(Matrix<double>[] input) {
            foreach(var layer in Layers) {
                input = layer.Propagate(input);
            }
            return input;
        }

        public override Matrix<double>[] Backpropagate(Matrix<double>[] error, double lnr) {
            for (int i = Layers.Count-1; i>-1;i--)
                error = Layers[i].Backpropagate(error, lnr);
            return error;
        }

        static public void SaveNetwork(Network network) {
            new BinaryFormatter().Serialize(new FileStream("savedNetwork.txt", FileMode.Create, FileAccess.Write), network);
        }

        static public Network LoadNetwork() {
            return (Network)new BinaryFormatter().Deserialize(new FileStream("savedNetwork.txt", FileMode.Open, FileAccess.Read));
        }
    }
}
