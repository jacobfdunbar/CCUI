#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <Windows.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int frameCountKeep = 0;

bool listenForKeyPress(int key) {
	return GetAsyncKeyState(key) & 0x8000;
}

int main(int argc, char ** argv) {
	cout << "Welcome to this demo!" << endl;
	this_thread::sleep_for(chrono::milliseconds(1000));
	cout << "Finding webcam..." << endl;
	VideoCapture vc(0);
	if (!vc.isOpened()) {
		cout << "Failed!" << endl;
		this_thread::sleep_for(chrono::milliseconds(2000));
		return -1;
	}
	cout << "Found!" << endl;
	cout << "Displaying webcam footage..." << endl;
	cout << "Press 'Esc' to quit." << endl;
	this_thread::sleep_for(chrono::milliseconds(500));
	auto begin = chrono::high_resolution_clock::now();
	int frameCount = 0;

	Mat gray, prevGray, image, frame;
	vector<Point2f> points[2];
	bool nightMode = false;
	bool needToInit = true;
	const int MAX_COUNT = 500;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	Point2f point;
	int hSpread = 20;
	int xBase = 20;
	int vSpread = 20;
	int yBase = 20;
	for (int i = 0; i < 23; i++) {
		for (int k = 0; k < 31; k++) {
			point = Point2f((float)(k * hSpread) + xBase, (float)(i * vSpread) + yBase);
			points[0].push_back(point);
		}
	}

	while (1) {
		vc.read(frame);
		if (frame.empty()) {
			break;
		}

		//Optical Flow
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);
		
		float overallMag = 0;
		float vecx = 0, vecy = 0;
		int movePoints = 0;

		if (nightMode) {
			image = Scalar::all(0);
		}

		if (needToInit) {
			//goodFeaturesToTrack(gray, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			//cornerSubPix(gray, points[0], subPixWinSize, Size(-1, -1), termcrit);
			needToInit = false;
		}
		else if (!points[0].empty()) {
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty()) {
				gray.copyTo(prevGray);
			}
			
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			float totalMagX = 0;
			float totalMagY = 0;
			for (i = 0; i < points[0].size(); i++)
			{
				circle(image, points[0][i], 1, Scalar(0, 0, 255), -1, 8);
				if (!status[i])
					continue;

				//points[1][k++] = points[1][i];
				double xdif = points[1][i].x - points[0][i].x;
				double ydif = points[1][i].y - points[0][i].y;
				float magnitude = sqrt(pow(xdif, 2) + pow(ydif, 2));
				if (magnitude >= 5) {
					circle(image, points[0][i], 3, Scalar(0, 255, 0), -1, 8);
					arrowedLine(
						image, 
						Point(points[0][i].x, points[0][i].y), 
						Point(points[1][i].x, points[1][i].y),
						Scalar(0, 0, 0),
						1, 
						8, 
						0);
					totalMagX += xdif;
					totalMagY += ydif;
					movePoints++;
				}
			}
			float magx = totalMagX / movePoints;
			float magy = totalMagY / movePoints;
			overallMag = sqrt(pow(abs(magx), 2)
				+ pow(abs(magy), 2));
			vecx = magx / overallMag;
			vecy = magy / overallMag;
			if (overallMag > 10) {
				arrowedLine(
					image,
					Point(image.cols / 2, image.rows / 2),
					Point(vecx * 100 + (image.cols / 2), vecy * 100 + (image.rows / 2)),
					Scalar(255, 0, 0),
					5,
					8,
					0);
			}
		}
		else {
			for (int i = 0; i < points[0].size(); i++) {
				circle(image, points[0][i], 3, Scalar(0, 255, 0), -1, 8);
			}
		}

		if (waitKey(10) == 27) break;
		frameCount++;
		auto end = chrono::high_resolution_clock::now();
		if (chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() > 1000) {
			frameCountKeep = frameCount;
			begin = chrono::high_resolution_clock::now();
			frameCount = 0;
		}
		Mat flip;
		cv::flip(image, flip, 1);
		cv::putText(
			flip,
			cv::String("FPS:" + std::to_string(frameCountKeep)),
			Point(500, 50),
			FONT_HERSHEY_COMPLEX_SMALL,
			0.8,
			Scalar(0, 0, 0),
			1,
			CV_AA
		);
		cv::putText(
			flip,
			String("Mag: " + std::to_string(overallMag)),
			Point(300, 50),
			FONT_HERSHEY_COMPLEX_SMALL,
			0.8,
			Scalar(0, 0, 0),
			1,
			CV_AA
		);
		cv::putText(
			flip,
			String("Vector: <" + std::to_string(vecx) + ", " + std::to_string(-vecy) + ">"),
			Point(300, 100),
			FONT_HERSHEY_COMPLEX_SMALL,
			0.8,
			Scalar(0, 0, 0),
			1,
			CV_AA
		);
		cv::imshow("Webcam", flip);
		cv::swap(prevGray, gray);
	}
}