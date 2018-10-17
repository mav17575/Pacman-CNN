using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PacMan_Conv.Activations;
using PacMan_Conv.LossFunctions;
using PacMan_Conv.NeuralNetwork;
using PacMan_Conv.NeuralNetwork.Layers;
using System.IO;
using System.Net;

namespace PacMan_Conv {
    class Program {
        /// <summary>
        /// only for test purpose
        /// </summary>
        /// <forDevs>
        /// code to test only in here
        /// </forDevs>
        /// <param name="args">The command-line arguments.</param>
        static void Main(string[] args) {
            //Run();
            Test();
        }

        public static void Test(){
            //Test.Test1();

            Random random = new Random();

            var convLayer1 = new ConvolutionLayer(1, 10, 5, new Sigmoid());
            var pool1 = new MaxPoolingLayer(4);
            var convLayer2 = new ConvolutionLayer(10, 20, 3, new Sigmoid());
            var pool2 = new MaxPoolingLayer(4);
            var denseLayer1 = new DenseLayer(31960, 70, new Sigmoid(), true);
            var denseLayer2 = new DenseLayer(70, 10, new Sigmoid(), false);

            var network = new Network(new MeanSquaredError(), convLayer1, pool1, convLayer2, pool2, denseLayer1, denseLayer2);

            var map = new Bitmap("dog.jpg");
            var input1 = new Matrix<double>[] { ImgUtil.GetPixelsGray(map) };


            Stopwatch watch1 = new Stopwatch();
            watch1.Start();
            network.Propagate(input1);
            watch1.Stop();
            Console.WriteLine("Propagate Time: " + watch1.ElapsedMilliseconds + "ms");
            for (int a1 = 0; a1 < 10; a1++)
                ImgUtil.ConvertToBitmap(convLayer1.Last_Output[a1], convLayer1.Last_Output[a1].ColumnCount, convLayer1.Last_Output[a1].RowCount, "/Users/timoluick/test" + a1 + ".jpg", "Sigmoid");



            FileStream ifsLabels = new FileStream("t10k-labels.idx1-ubyte", FileMode.Open); // test labels
            FileStream ifsImages = new FileStream("t10k-images.idx3-ubyte", FileMode.Open); // test images

            BinaryReader brLabels = new BinaryReader(ifsLabels);
            BinaryReader brImages = new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            int images = 10000;
            Matrix<double>[] input = new Matrix<double>[images];
            Matrix<double>[] labels = new Matrix<double>[images];



            /*// each test image
            for (int di = 0; di < 10000; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                    }
                }

                byte lbl = brLabels.ReadByte();
            } // each image*/


            Stopwatch watch = new Stopwatch();
            watch.Start();
            for (int i = 0; i < images; i++)
            {
                input[i] = DenseMatrix.Create(28, 28, 0);
                labels[i] = DenseMatrix.Create(10, 1, 0);
                for (int x = 0; x < 28; x++)
                {
                    for (int y = 0; y < 28; y++)
                        input[i][x, y] = (double)brImages.ReadByte() / 255;
                }
                byte index = brLabels.ReadByte();
                labels[i][index, 0] = 1;
            }
            watch.Stop();
            var a = input[0];
            /*int n = input.Length;
            while (n > 1)
            {
                int k = random.Next(n--);
                var temp = input[n];
                input[n] = input[k];
                input[k] = temp;
            }*/
            Console.WriteLine("Time to generate images: " + watch.ElapsedMilliseconds + "ms");

            for (int b = 0; b < 100; b++)
            {
                double acc = 0;
                int a3 = 0;
                for (int i = 0; i < images; i++)
                {

                    var output = network.Propagate(new Matrix<double>[] { input[i] });

                    if (i % 100 == 0)
                    {
                        for (int a1 = 0; a1 < 10; a1++)
                            ImgUtil.ConvertToBitmap(convLayer1.Last_Output[a1], convLayer1.Last_Output[a1].ColumnCount, convLayer1.Last_Output[a1].RowCount, "/Users/timoluick/test" + a1 + ".jpg", "Sigmoid");
                    }

                    var error1 = labels[i] - output[0];
                    var error = new Matrix<double>[1];
                    error[0] = error1;
                    network.Backpropagate(error, .001);

                    if (i % 10 == -1)
                    {
                        Console.WriteLine("output:" + output[0].Column(0).MaximumIndex());
                        Console.WriteLine("actual label:" + labels[i].Column(0).MaximumIndex());
                    }
                    a3++;
                    if (output[0].Column(0).MaximumIndex() == labels[i].Column(0).MaximumIndex())
                        acc++;
                    if (i % 1 == 0)
                    {
                        Console.Clear();
                        Console.WriteLine("Epoch: " + (b + 1) + "  " + i + "/" + images + ";   Accuracy:" + Math.Round((acc / i * 100), 0) + "%   Hits:" + acc);
                        a3 = 0;
                    }
                }
            }



            /*MathNet.Numerics.Control.UseMultiThreading();

            var input = ImgUtil.GetPixelsRGB(new Bitmap("cnnimg.jpg"));//DenseMatrix.CreateRandom(400, 400, RandomUtil.UnitUniform);
            //var input = new Matrix<double>[] { m };

            var c1 = new ConvolutionLayer(3, 6, 3, new Sigmoid());
            var m1 = new MaxPoolingLayer(3);
            var c2 = new ConvolutionLayer(6, 12, 2, new Sigmoid());
            var m2 = new MaxPoolingLayer(3);
            var c3 = new ConvolutionLayer(12, 12, 5, new Sigmoid());
            var d1 = new DenseLayer(32376, 1000, new Sigmoid(), true);
            var d2 = new DenseLayer(1000, 100, new Sigmoid(), false);
            var d3 = new DenseLayer(100, 1, new Sigmoid(), false);

            var network = new Network(new MeanSquaredError(), new List<Layer> {c1, m1, c2, m2, c3, d1, d2, d3});

            /*Console.WriteLine(network.Propagate(input)[0]);
            Network.SaveNetwork(network);
            network = Network.LoadNetwork();
            Console.WriteLine(network.Propagate(input)[0]);
            */

            /*Stopwatch watch = new Stopwatch();
            watch.Start();

            for (int a = 0; a < 60; a++) {
                Stopwatch wat = new Stopwatch();
                wat.Start();
                var result = network.Propagate(input);
                wat.Stop();
                Console.WriteLine("-------------#" + a + "--------------");
                Console.WriteLine("Time Propagate: " + wat.ElapsedMilliseconds + "ms");
                Console.WriteLine("");
                Console.WriteLine(result[0]);
                Console.WriteLine("");
                wat.Restart();
                wat.Start();
                var backward_result = network.Backpropagate(network.Target.Target(new Matrix<double>[] { DenseMatrix.Create(1, 1, 1) }, result), 0.001);
                wat.Stop();
                Console.WriteLine("Time Backpropagate: " + wat.ElapsedMilliseconds + "ms");
                Console.WriteLine("-----------------------------");
                for (int a1 = 0; a1 < 6; a1++)
                    ImgUtil.ConvertToBitmap(c1.Last_Output[a1], c1.Last_Output[a1].ColumnCount, c1.Last_Output[a1].RowCount, "/Users/timoluick/test" + a1 + ".jpg", "LeakyRelu");
            }
            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);
            Console.ReadLine();*/
        }

        static public void Run() {
            MathNet.Numerics.Control.UseMultiThreading();

            var convLayer1 = new ConvolutionLayer(3, 4, 3, new Sigmoid());
            var poolLayer1 = new MaxPoolingLayer(2);
            var convLayer2 = new ConvolutionLayer(4, 8, 5, new Sigmoid());
            var poolLayer2 = new MaxPoolingLayer(3);
            var convLayer3 = new ConvolutionLayer(8, 12, 5, new Sigmoid());
            var denseLayer1 = new DenseLayer(76464, 5000, new Sigmoid(), true);
            var denseLayer2 = new DenseLayer(5000, 500, new Sigmoid(), false);
            var denseLayer3 = new DenseLayer(500, 50, new Sigmoid(), false);
            var denseLayer4 = new DenseLayer(50, 4, new Sigmoid(), false);

            var network = new Network(new MeanSquaredError(), convLayer1, poolLayer1, convLayer2, poolLayer2, convLayer3, denseLayer1, denseLayer2, denseLayer3, denseLayer4);


            Bitmap bitmap;
            using (WebClient webClient = new WebClient())
            {
                // http://192.168.0.80/api/cam/lvgetimg?d=2018-09-27T16:49:28153
                webClient.Headers.Add(HttpRequestHeader.Cookie, "acid=3671");
                while(true)
                {
                    DateTime dt = DateTime.Now;
                    String date = dt.ToString("yyyy-MM-dd");
                    String day = dt.ToString("HH:mm:ssFFF");
                    String d = date + "T" + day;
                    // http://192.168.0.80/api/cam/lvgetimg?d=2018-10-11T15:46:38113
                    var response = webClient.DownloadData("http://192.168.0.80/api/cam/lvgetimg?d=" + d);
                    bitmap = ImgUtil.ByteToImage(response);
                    var input = ImgUtil.GetPixelsRGB(bitmap);
                    network.Propagate(input);
                    for (int a = 0; a < 12; a++)
                        ImgUtil.ConvertToBitmap(convLayer3.Last_Output[a], convLayer3.Last_Output[a].ColumnCount, convLayer3.Last_Output[a].RowCount, "/Users/timoluick/test" + a + ".jpg", "Sigmoid");
                    Console.WriteLine("done");
                }
            }

        }

    }
}
