
namespace PacMan_Conv.Activations {
    public class LeakyReLu : Activation {

        public override double Activate(double x) {
            return (x < 0) ? x * 0.01 : x;
        }

        public override double Derivative(double y) {
            return (y < 0) ? y / 0.01 : y;
        }
    }
}
