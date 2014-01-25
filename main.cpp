/*
 * EEL6562 Project
 * Real Time Object Recognition using SURF
 *
 *  Created on: Nov 15, 2013
 *      Author: Frank
 */

//Include statements
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

//Name spaces used
using namespace cv;
using namespace std;

int main()
{
	//turn performance analysis functions on if testing = true
	bool testing=false;
	double t; //timing variable

	//load training image
	Mat object = imread ("C:/School/Image Processing/book.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (!object.data){
		cout<<"Can't open image";
		return -1;
	}
	namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);

	//SURF Detector, and descriptor parameters
	int minHess=3000;
	vector<KeyPoint> kpObject, kpImage;
	Mat desObject, desImage;


	//Performance measures calculations for report
	if (testing)
	{
		cout<<object.rows<<" "<<object.cols<<endl;

		//calculate integral image
		Mat iObject;
		integral(object, iObject);
		imshow("Good Matches", iObject);
		imwrite("C:/School/Image Processing/IntegralImage.jpg", iObject);
		cvWaitKey(0);

		//calculate number of interest points, computation time as f(minHess)
		int minHessVector[]={100, 500, 1000, 1500, 2000, 2500, 3000,
									3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500,
									8000, 8500, 9000, 9500, 10000};
		int minH;
		std::ofstream file;
		file.open("C:/School/Image Processing/TimingC.csv", std::ofstream::out);
		for (int i=0; i<20; i++)
		{
			minH=minHessVector[i];
			t = (double)getTickCount();
			SurfFeatureDetector detector(minH);
			detector.detect(object, kpObject);
			t = ((double)getTickCount() - t)/getTickFrequency();
			file<<minHess<<","<<kpObject.size()<<","<<t<<",";
			cout<<t<<" "<<kpObject.size()<<" "<<desObject.size()<<endl;

			t = (double)getTickCount();
			SurfDescriptorExtractor extractor;
			extractor.compute(object, kpObject, desObject);
			t = ((double)getTickCount() - t)/getTickFrequency();
			file<<t<<endl;
		}
		file.close();

	//Display keypoints on training image
	Mat interestPointObject=object;
	for (unsigned int i=0; i<kpObject.size();i++)
	{
		if(kpObject[i].octave)
		{
			circle(interestPointObject,kpObject[i].pt,kpObject[i].size,0);
			string octaveS;
			switch(kpObject[i].octave)
			{
			case 0:
				octaveS="0";
				break;
			case 1:
				octaveS='1';
				break;
			case 2:
				octaveS='2';
				break;
			default:
				break;

			}
			putText(interestPointObject, octaveS, kpObject[i].pt,
				    FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(0,0,250), 1, CV_AA);
		}

	}
	imshow("Good Matches",interestPointObject);
	imwrite("C:/School/Image Processing/bookIP2.jpg", interestPointObject);
	cvWaitKey(0);
	}


	//SURF Detector, and descriptor parameters, match object initialization
	minHess=2000;
	SurfFeatureDetector detector(minHess);
	detector.detect(object, kpObject);
	SurfDescriptorExtractor extractor;
	extractor.compute(object, kpObject, desObject);
	FlannBasedMatcher matcher;

	//Initialize video and display window
	VideoCapture cap(1);  //camera 1 is webcam
	if (!cap.isOpened()) return -1;

	//Object corner points for plotting box
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0);
	obj_corners[1] = cvPoint( object.cols, 0 );
    obj_corners[2] = cvPoint( object.cols, object.rows );
    obj_corners[3] = cvPoint( 0, object.rows );

    //video loop
    char escapeKey='k';
    double frameCount = 0;
    float thresholdMatchingNN=0.7;
    unsigned int thresholdGoodMatches=4;
    unsigned int thresholdGoodMatchesV[]={4,5,6,7,8,9,10};

    for (int j=0; j<7;j++){
    	thresholdGoodMatches=thresholdGoodMatchesV[j];
    	//thresholdGoodMatches=8;
    	cout<<thresholdGoodMatches<<endl;

    if(true)
    {
    	t = (double)getTickCount();
    }

    while (escapeKey != 'q')
    {
    	frameCount++;
    	Mat frame;
    	Mat image;
    	cap>>frame;
    	cvtColor(frame, image, CV_RGB2GRAY);

    	Mat des_image, img_matches, H;
    	vector<KeyPoint> kp_image;
    	vector<vector<DMatch > > matches;
    	vector<DMatch > good_matches;
    	vector<Point2f> obj;
    	vector<Point2f> scene;
    	vector<Point2f> scene_corners(4);

    	detector.detect( image, kp_image );
    	extractor.compute( image, kp_image, des_image );
    	matcher.knnMatch(desObject, des_image, matches, 2);

    	                            for(int i = 0; i < min(des_image.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
    	                            {
    	                            	if((matches[i][0].distance < thresholdMatchingNN*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
    	                                {
    	                                    good_matches.push_back(matches[i][0]);
    	                                }
    	                            }

    	                            //if (good_matches.size()<1)
    	                            //	good_matches.resize(0,cv::DMatch);

    	                            //Draw only "good" matches
    	                            drawMatches( object, kpObject, image, kp_image, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


    	                            if (good_matches.size() >= thresholdGoodMatches)
    	                            {

    	                            	//Display that the object is found
    	                            	putText(img_matches, "Object Found", cvPoint(10,50),FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0,0,250), 1, CV_AA);
    	                                for(unsigned int i = 0; i < good_matches.size(); i++ )
    	                                {
    	                                    //Get the keypoints from the good matches
    	                                    obj.push_back( kpObject[ good_matches[i].queryIdx ].pt );
    	                                    scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
    	                                }

    	                                H = findHomography( obj, scene, CV_RANSAC );

    	                                perspectiveTransform( obj_corners, scene_corners, H);

    	                                //Draw lines between the corners (the mapped object in the scene image )
    	                                line( img_matches, scene_corners[0] + Point2f( object.cols, 0), scene_corners[1] + Point2f( object.cols, 0), Scalar(0, 255, 0), 4 );
    	                                line( img_matches, scene_corners[1] + Point2f( object.cols, 0), scene_corners[2] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
    	                                line( img_matches, scene_corners[2] + Point2f( object.cols, 0), scene_corners[3] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
    	                                line( img_matches, scene_corners[3] + Point2f( object.cols, 0), scene_corners[0] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
    	                            }
    	                            else
    	                            {
    	                            	putText(img_matches, "", cvPoint(10,50), FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(0,0,250), 1, CV_AA);
    	                            }

    	                            //Show detected matches
    	                            imshow( "Good Matches", img_matches );
    	                            escapeKey=cvWaitKey(10);
    	                            //imwrite("C:/School/Image Processing/bookIP3.jpg", img_matches);

    	                            if(frameCount>10)
    	                            	escapeKey='q';


    }

    //average frames per second
    if(true)
    {
    	t = ((double)getTickCount() - t)/getTickFrequency();
    	cout<<t<<" "<<frameCount/t<<endl;
    	cvWaitKey(0);
    }

    frameCount=0;
    escapeKey='a';
    }

    //Release camera and exit
    cap.release();
	return 0;
}
