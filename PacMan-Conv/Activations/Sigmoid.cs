using System;
namespace PacMan_Conv.Activations {
    public class Sigmoid : Activation {

        public override double Activate(double x) {
            return 1 / (1 + Math.Exp(-x));
        }

        public override double DeActivate(double y) {
            return y * (1 - y);
        }
    }
}
