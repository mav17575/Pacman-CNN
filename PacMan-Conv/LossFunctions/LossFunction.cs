using MathNet.Numerics.LinearAlgebra;

namespace PacMan_Conv.LossFunctions {
    abstract public class LossFunction {
        public abstract Matrix<double>[] Target(params Matrix<double>[][] inputs);
    }
}
