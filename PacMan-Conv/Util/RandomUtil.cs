using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PacMan_Conv
{
    class RandomUtil
    {
        //Soll RNGs bieten. Wird erweitert...

        public static ContinuousUniform UnitUniform = new ContinuousUniform(-1,1);
        public static Random NormalRandom = new Random();

        public static ContinuousUniform CreateUniform(double lower, double upper)
        {
            return new ContinuousUniform(lower,upper);
        }
    }
}
