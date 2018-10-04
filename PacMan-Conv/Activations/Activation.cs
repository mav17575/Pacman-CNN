using System;

namespace PacMan_Conv.Activations {
    abstract public class Activation {

        abstract public double Activate(double x);
        abstract public double DeActivate(double y);
    }
}
