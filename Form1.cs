using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Flann;
using Emgu.CV.Cuda;
using System.Runtime.InteropServices;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
//using ClosedXML.Excel;
using Emgu.CV.UI;
//using EmgucvDemo.Models;
using Emgu.CV.ML;
using Emgu.CV.ML.MlEnum;
using System.IO;
using System.Runtime.InteropServices.WindowsRuntime;
using Emgu.CV.Face;
namespace ElementH
{
    public partial class Form1 : Form
    {
       // bool startRecording;
        VideoCapture cap;
        Image<Bgr, byte> imgInput;
       // Image<Bgr, byte> imagePB1;
        public Form1()
        {
            InitializeComponent();
            int imagesGrabbed = 0;
            //cap.Start();
            imgInput = new Image<Bgr, byte>(new Size(800, 600));
            try
            {
                cap = new VideoCapture(0);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            cap.ImageGrabbed += ProcessFrame;
        }

        private void zamknijToolStripMenuItem_Click(object sender, EventArgs e)
        {
            
            if (MessageBox.Show("Jesteś pewny, że chcesz zamknąć?","Wiadomość systemu", MessageBoxButtons.YesNo) == DialogResult.Yes)
            {
                this.Close();
            }
        }

        private void otwórzToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog dialog = new OpenFileDialog();
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    imgInput = new Image<Bgr, byte>(dialog.FileName);
                    pictureBox1.Image = imgInput.Bitmap;
                }
            }
            catch(Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void cannyToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (imgInput == null)
            {
                return;
            }
            Image<Gray, byte> imgCanny = new Image<Gray, byte>(imgInput.Width, imgInput.Height, new Gray(0));
            imgCanny = imgInput.Canny(50, 20);
            pictureBox2.Image = imgCanny.Bitmap;
        }

        private void sobelToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (imgInput == null)
            {
                return;
            }
            Image<Gray, byte> imgGray = imgInput.Convert<Gray, byte>();
            Image<Gray, float> imgSobel = new Image<Gray,float>(imgInput.Width, imgInput.Height, new Gray(0));
            imgSobel= imgGray.Sobel(1,1,3);
            pictureBox2.Image = imgSobel.Bitmap;
        }

        private void laplacianToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (imgInput == null)
            {
                return;
            }
            Image<Gray, byte> imgGray = imgInput.Convert<Gray, byte>();
            Image<Gray, float> imgLaplacian = new Image<Gray, float>(imgInput.Width, imgInput.Height, new Gray(0));
            imgLaplacian = imgGray.Laplace(7);
            pictureBox2.Image = imgLaplacian.Bitmap;
        }

        private void kameraToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }

        private void startToolStripMenuItem_Click(object sender, EventArgs e)
        {
            //startRecording = !startRecording;
            
            cap.Start();
           
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            Mat m=new Mat();
            cap.Read(m);
            if(m != null)
            {
                imgInput = m.ToImage<Bgr, byte>();
                pictureBox1.Image = imgInput.Bitmap;
            }
            
        }

        private void stopToolStripMenuItem_Click(object sender, EventArgs e)
        {
            cap.Stop();
            imgInput.SetZero();
            pictureBox1.Image = imgInput.Bitmap;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (imgInput == null)
            {
                return;
            }

            try
            {
                var temp = imgInput.SmoothGaussian(5).Convert<Gray, byte>().ThresholdBinaryInv(new Gray(230), new Gray(255));

                VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
                Mat m = new Mat();

                CvInvoke.FindContours(temp, contours, m, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

                for (int i = 0; i < contours.Size; i++)
                {
                    double perimeter = CvInvoke.ArcLength(contours[i], true);
                    VectorOfPoint approx = new VectorOfPoint();
                    CvInvoke.ApproxPolyDP(contours[i], approx, 0.04 * perimeter, true);

                    CvInvoke.DrawContours(imgInput, contours, i, new MCvScalar(135, 0, 25), 2);

                    //moments  center of the shape

                    var moments = CvInvoke.Moments(contours[i]);
                    int x = (int)(moments.M10 / moments.M00);
                    int y = (int)(moments.M01 / moments.M00);

                    if (approx.Size == 3)
                    {
                        CvInvoke.PutText(imgInput, "Triangle", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
                    }

                    if (approx.Size == 4)
                    {
                        Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);

                        double ar = (double)rect.Width / rect.Height;

                        if (ar >= 0.95 && ar <= 1.05)
                        {
                            CvInvoke.PutText(imgInput, "Square", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
                        }
                        else
                        {
                            CvInvoke.PutText(imgInput, "Rectangle", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
                        }

                    }

                    if (approx.Size == 6)
                    {
                        CvInvoke.PutText(imgInput, "Hexagon", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
                    }


                    if (approx.Size == 6)
                    {
                        CvInvoke.PutText(imgInput, "Circle", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
                    }
                    if (approx.Size > 7)
                    {
                        CvInvoke.PutText(imgInput, "ElementH", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
                    }
                    pictureBox2.Image = imgInput.Bitmap;

                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            try
            {
                if (pictureBox1.Image == null) return;

                


                var gray = imgInput.Convert<Gray, byte>()
                    .ThresholdBinaryInv(new Gray(240), new Gray(255));

                // contours
                VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
                Mat h = new Mat();

                CvInvoke.FindContours(gray, contours, h, Emgu.CV.CvEnum.RetrType.External
                    , Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);


                VectorOfPoint approx = new VectorOfPoint();

                Dictionary<int, double> shapes = new Dictionary<int, double>();

                for (int i = 0; i < contours.Size; i++)
                {
                    approx.Clear();
                    double perimeter = CvInvoke.ArcLength(contours[i], true);
                    CvInvoke.ApproxPolyDP(contours[i], approx, 0.04 * perimeter, true);
                    double area = CvInvoke.ContourArea(contours[i]);

                    if (approx.Size > 6)
                    {
                        shapes.Add(i, area);
                    }
                }


                if (shapes.Count > 0)
                {
                    var sortedShapes = (from item in shapes
                                        orderby item.Value ascending
                                        select item).ToList();

                    for (int i = 0; i < sortedShapes.Count; i++)
                    {
                        CvInvoke.DrawContours(imgInput, contours, sortedShapes[i].Key, new MCvScalar(0, 0, 255), 2);
                        var moments = CvInvoke.Moments(contours[sortedShapes[i].Key]);
                        int x = (int)(moments.M10 / moments.M00);
                        int y = (int)(moments.M01 / moments.M00);

                        CvInvoke.PutText(imgInput, (i + 1).ToString(), new Point(x, y), Emgu.CV.CvEnum.FontFace.HersheyTriplex, 1.0,
                            new MCvScalar(0, 0, 255), 2);
                        CvInvoke.PutText(imgInput, sortedShapes[i].Value.ToString(), new Point(x, y - 30), Emgu.CV.CvEnum.FontFace.HersheyTriplex, 1.0,
                            new MCvScalar(0, 0, 255), 2);
                    }

                }

                pictureBox1.Image = imgInput.ToBitmap();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
    }
}
