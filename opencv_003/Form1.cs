using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;







using System.IO;
using OpenCvSharp.XFeatures2D;

namespace opencv_003
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        VideoCapture videoCapture;
        OpenCvSharp.Face.FisherFaceRecognizer model1;
        OpenCvSharp.Face.EigenFaceRecognizer model2;

        List<ImageInfo> Face_images_info = new List<ImageInfo>();





        internal class ImageInfo
        {
            public Mat Image { set; get; }
            public int ImageGroupId { set; get; }
            public int ImageId { set; get; }
            public String person_name { set; get; }
        }



        private void Form1_Load(object sender, EventArgs e)
        {
            for (int deviceNum = 0; 99999 > deviceNum; deviceNum++)
            {
                videoCapture = new VideoCapture(deviceNum);
                if (videoCapture.IsOpened() == true)
                {
                    log_write("카메라 연결 성공 DeviceID : " + Convert.ToString(deviceNum));
                    break;
                }
            }

            videoCapture.Set(CaptureProperty.FrameWidth, 640);
            videoCapture.Set(CaptureProperty.FrameHeight, 480);

            log_write("videoCapture.FrameWidth:" + Convert.ToString(videoCapture.FrameWidth));
            log_write("videoCapture.FrameHeight:" + Convert.ToString(videoCapture.FrameHeight));

            FaceTrain_Step2();
        }

        private void Timer1_Tick(object sender, EventArgs e)
        {
            if (videoCapture.IsOpened() == false)
            {
                log_write("카메라 연결 안됨");
                timer1.Stop();
            }


            Mat frame_img_source_01 = new Mat();
            videoCapture.Read(frame_img_source_01);


            Mat searchFace = new Mat();
            Mat temp_image = new Mat();

            Rect[] faces = FaceDetect(frame_img_source_01);

            for (int faceCnt = 0; faceCnt < faces.Length; faceCnt++)
            {

                //  log_write("faces:" + Convert.ToString(faces[faceCnt]));
                if (faces[faceCnt].Width > 100)
                {
                    Cv2.Resize(frame_img_source_01[faces[faceCnt]], searchFace, new OpenCvSharp.Size(160, 160));
                    byte[] imageBytes = searchFace.ToBytes(".bmp");
                    searchFace = Mat.FromImageData(imageBytes, ImreadModes.Grayscale);

                    var predictedGroupId = 0;
                    if (model1 != null)
                    {
                        predictedGroupId = model1.Predict(searchFace);
                    }

                    if (predictedGroupId == 0)
                    {
                        log_write("predictedGroupId:" + Convert.ToString("확인불가"));
                        log_write("predictedGroupId:" + Convert.ToString(predictedGroupId));
                        Cv2.DestroyAllWindows();
                    }
                    else
                    {
                        log_write("predictedGroupId:" + Convert.ToString(predictedGroupId));
                        log_write("성명:" + Convert.ToString(Face_images_info.FirstOrDefault(w => w.ImageGroupId == predictedGroupId).person_name));

                        image_maatching(searchFace, Face_images_info.FirstOrDefault(w => w.ImageGroupId == predictedGroupId).Image);
                        
                    }

                    Cv2.Rectangle(frame_img_source_01, faces[faceCnt], Scalar.YellowGreen, 2);
                }
                
            }

            Cv2.Flip(frame_img_source_01, frame_img_source_01, FlipMode.Y);
            pictureBox1.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame_img_source_01);

        }


        private void image_maatching(Mat img1, Mat img2)
        {

            Cv2.ImShow("Matches1", img1);
            Cv2.ImShow("Matches2", img2);
            
            var detector = SURF.Create(hessianThreshold: 300,4,2,true,false); //A good default value could be from 300 to 500, depending from the image contrast.

            KeyPoint[] keypoints1 = null;
            KeyPoint[] keypoints2 = null;

            Mat descriptors1 = new Mat();
            Mat descriptors2 = new Mat();

            detector.DetectAndCompute(img1, null, out keypoints1, descriptors1);
            detector.DetectAndCompute(img2, null, out keypoints2, descriptors2);

            var matcher = new BFMatcher();
            var matches = matcher.Match(descriptors1, descriptors2);

            float max_dist = 50;
            int cntSuccessPoint = 0;
            for (int i = 0; i < matches.Length; i++)
            {
                log_write("matches[i].Distance:" + Convert.ToString(max_dist) +"--"+ Convert.ToString(matches[i].Distance));
                if(( matches[i].Distance * 100 ) < max_dist)
                {
                    cntSuccessPoint = cntSuccessPoint + 1;
                }
            }// end for

            double rate = (cntSuccessPoint * 100) / matches.Length;
            log_write("유사율:" + Convert.ToString(rate)  +"---"+ Convert.ToString(cntSuccessPoint) + "/" + Convert.ToString(matches.Length) );

            var imgMatches = new Mat();
            Cv2.DrawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
            Cv2.ImShow("Matches3", imgMatches);

        }



        private void FaceTrain_Step1(Mat source_image, Rect[] faces)
        {
            timer1.Stop();
            
            byte[] imageBytes = source_image.ToBytes(".bmp");
            source_image = Mat.FromImageData(imageBytes, ImreadModes.Grayscale);

            Mat temp_image = new Mat();

            String saveBitmapPath = null;
            String DirName = textBox1.Text;
            Bitmap tempBitMap = null;

            OpenCvSharp.Size size = new OpenCvSharp.Size(30,30);

            for (int faceCnt = 0; faceCnt < faces.Length; faceCnt++)
            {
                Cv2.Resize(source_image[faces[faceCnt]], temp_image, new OpenCvSharp.Size(160, 160));

                tempBitMap = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(temp_image);
                pictureBox1.Image = tempBitMap;
                log_write("faceCnt_images.Add:" + Convert.ToString(faceCnt));

                saveBitmapPath = "C://Users//dev-yym//source//repos//opencv_003//Face_image";
                saveBitmapPath = saveBitmapPath + "//" + Convert.ToString(DirName) + "//";

                DirectoryInfo di = new DirectoryInfo(saveBitmapPath);

                if(di.Exists == false)
                {
                    di.Create();
                }

                saveBitmapPath = saveBitmapPath + "//";
                saveBitmapPath = saveBitmapPath + Convert.ToString(DateTime.Now.Year);
                saveBitmapPath = saveBitmapPath + Convert.ToString(DateTime.Now.Month);
                saveBitmapPath = saveBitmapPath + Convert.ToString(DateTime.Now.Day);
                saveBitmapPath = saveBitmapPath + Convert.ToString(DateTime.Now.Hour);
                saveBitmapPath = saveBitmapPath + Convert.ToString(DateTime.Now.Minute);
                saveBitmapPath = saveBitmapPath + Convert.ToString(DateTime.Now.Second);
                saveBitmapPath = saveBitmapPath + "_" + Convert.ToString(faceCnt) + ".bmp";
                log_write("saveBitmapPath:" + Convert.ToString(saveBitmapPath));

                tempBitMap.Save(saveBitmapPath);
            }


            timer1.Start();
        }



        private void FaceTrain_Step2()
        {
            model1 = null;
            model2 = null;
            Face_images_info.Clear();

            var imageId = 0;
            var groupId = 0;
            String saveBitmapPath = "C://Users//dev-yym//source//repos//opencv_003//Face_image";

            foreach (var dir in new DirectoryInfo(saveBitmapPath).GetDirectories())
            {
                
                foreach (var imageFile in dir.GetFiles("*.bmp"))
                {
                    Face_images_info.Add(new ImageInfo
                    {
                        Image = new Mat(imageFile.FullName,  ImreadModes.Grayscale),
                        ImageId = imageId++,
                        ImageGroupId = groupId++,
                        person_name = dir.Name
                    });
                }

            }

            

            if ( imageId > 0 )
            {
                
                model1 = OpenCvSharp.Face.FisherFaceRecognizer.Create();
                model1.Train(Face_images_info.Select(x => x.Image), Face_images_info.Select(x => x.ImageGroupId));
                log_write("OpenCvSharp.Face.FisherFaceRecognizer - 학습");

                
                model2 = OpenCvSharp.Face.EigenFaceRecognizer.Create();
                model2.Train(Face_images_info.Select(x => x.Image), Face_images_info.Select(x => x.ImageGroupId));
                log_write("OpenCvSharp.Face.EigenFaceRecognizer - 학습");




            }
            //:Mat[1 * 20 * CV_32SC1, IsContinuous = True, IsSubmatrix = False, Ptr = 0x169dddf0, Data = 0x16a5e640]


        }




        private Rect[] FaceDetect(Mat source_image)
        {   

            // 그레이 스케일
            Mat frame_img_gray = new Mat();
            byte[] imageBytes = source_image.ToBytes(".bmp");
            //byte[] imageBytes = frame_img_FlipY.ToBytes(".bmp");
            frame_img_gray = Mat.FromImageData(imageBytes, ImreadModes.Grayscale);

            Cv2.EqualizeHist(frame_img_gray, frame_img_gray);
            var cascade = new CascadeClassifier("C://Users//dev-yym//source//repos//opencv_003//FaceML_Data//haarcascade_frontalface_alt.xml");
            //var nestedCascade = new CascadeClassifier("C://Users//dev-yym//source//repos//opencv_002//FaceML_Data//haarcascade_eye_tree_eyeglasses.xml");

            var faces = cascade.DetectMultiScale(
                image: frame_img_gray,
                scaleFactor: 1.1,
                minNeighbors: 2,
                flags: HaarDetectionType.DoRoughSearch | HaarDetectionType.ScaleImage,
                minSize: new OpenCvSharp.Size(30, 30)
                );
            return faces;
        }



        private void log_write(String logText)
        {
            log_1.Text = Convert.ToString(DateTime.Now) + "-" + logText + "\r\n" + log_1.Text;

        }

        private void Button1_Click_1(object sender, EventArgs e)
        {
            log_1.Text = null;
            log_write("START BUTTON CLICK");

            timer1.Stop();
            timer1.Interval = 50; //스케쥴 간격 설정
            timer1.Start(); //타이머를 가동시킨다.

            Cv2.DestroyAllWindows();
        }

        private void Button2_Click(object sender, EventArgs e)
        {
            log_write("STOP BUTTON CLICK");
            timer1.Stop();
        }

        private void Button3_Click(object sender, EventArgs e)
        {

            Mat source_image = new Mat();
            videoCapture.Read(source_image);
            Rect[] faces = FaceDetect(source_image);

            FaceTrain_Step1(source_image, faces);


            

        }

        private void Button4_Click(object sender, EventArgs e)
        {
            FaceTrain_Step2();
        }
    }
}




            



            


            
