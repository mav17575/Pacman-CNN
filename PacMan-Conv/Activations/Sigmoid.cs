using System;
namespace PacMan_Conv.Activations {
    [Serializable]
    //sigmoid activation function
    public class Sigmoid : Activation {

        /// <returns>value between 0 and 1</returns>
        public override double Activate(double x) {
            return 1 / (1 + Math.Exp(-x));
        }

        /// <summary>
        /// derivative of sigmoid
        /// </summary>
        /// <returns>derivative of y</returns>
        public override double Derivative(double y) {
            return y * (1 - y);
        }
    }
}
