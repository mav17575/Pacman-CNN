using System;

namespace PacMan_Conv.Activations {
    [Serializable]
    //struture for every activation function
    abstract public class Activation {

        abstract public double Activate(double x);
        abstract public double Derivative(double y);
    }
}
