using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PacMan_Conv
{
    class ImgUtil
    {
        public static double[] GetPixelsGrayArr(Bitmap img)
        {
            double[] res = new double[img.Width * img.Height];
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    System.Drawing.Color c = img.GetPixel(x, y);
                    double g = (c.R + c.G + c.B) / (765.0);
                    g = Math.Round(g, 3);
                    res[x + y * img.Width] = g;
                }
            }
            return res;
        }

        /// <summary>
        /// Converts an Image to a grayscale Matrix 
        /// </summary>
        /// <returns>matrix with gray values between 0 and 1</returns>
        /// <param name="img">Image as Bitmap</param>
        public static Matrix<double> GetPixelsGray(Bitmap img)
        {
            Matrix<double> res = new DenseMatrix(img.Height, img.Width);
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    System.Drawing.Color c = img.GetPixel(x, y);
                    double g = (c.R + c.G + c.B) / (765.0);
                    g = Math.Round(g, 3);
                    res[y, x] = g;
                }
            }
            return res;
        }
        public static Matrix<double>[] GetPixelsRGB(Bitmap img)
        {
            Matrix<double>[] res = new Matrix<double>[3];
            res[0] = new DenseMatrix(img.Height, img.Width);
            res[1] = new DenseMatrix(img.Height, img.Width);
            res[2] = new DenseMatrix(img.Height, img.Width);
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    System.Drawing.Color c = img.GetPixel(x, y);
                    double R = c.R / 255.0;
                    double G = c.G / 255.0;
                    double B = c.B / 255.0;
                    res[0][y, x] = R;
                    res[1][y, x] = G;
                    res[2][y, x] = B;
                }
            }
            return res;
        }

        public static Bitmap ByteToImage(byte[] data)
        {
            Bitmap bmp;
            using (var ms = new MemoryStream(data))
            {
                bmp = new Bitmap(ms);
            }
            return bmp;
        }

        public static Bitmap ConvertToBitmap(Matrix<double> m,int w, int h, String filename, String activation)
        {
            Bitmap conv = new Bitmap(w, h);
            SetPixels(conv, m, activation);
            conv.Save(filename);
            return conv;
        }

        public static void SetPixels(Bitmap img, Matrix<double> c, String activation)
        {
            //Debug.WriteLine(c.At(0, 0));
            double h = c.Enumerate().Max();
            double l = c.Enumerate().Min();
            l = l < 0 ? Math.Abs(l) : 0;
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    int val = 0;
                    switch (activation)
                    {
                        case "LeakyRelu":
                            val = (int)(((c.At(y, x) + l) / (h+l)) * 255.0);
                            break;
                        case "Sigmoid":
                            val = (int)(c.At(y, x) * 255.0);
                            break;
                        case "Tanh":
                            val = (int)(c.At(y, x) * 125.0) + 125;
                            break;
                    }
                    img.SetPixel(x, y, System.Drawing.Color.FromArgb(255, val, val, val));
                }
            }
        }

        public static Bitmap LoadBitmap(String path)
        {
            return new Bitmap(path);
        }

        public static Bitmap Random(int w, int h)
        {
            Bitmap res = new Bitmap(w,h);
            for (int y = 0; y<h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int r = RandomUtil.NormalRandom.Next(0,255);
                    int g = RandomUtil.NormalRandom.Next(0,255);
                    int b = RandomUtil.NormalRandom.Next(0,255);

                    res.SetPixel(x,y,Color.FromArgb(1,r,g,b));
                }
            }
            return res;
        }
    }
}
