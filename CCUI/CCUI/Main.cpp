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
int width = 31;
int height = 23;

const int NUM_CLUSTERS = 2;

Scalar clusterColors[] =
{
	Scalar(255, 255, 255),
	Scalar(0, 0, 255),
	Scalar(255, 0, 0),
	Scalar(0, 255, 0),
	Scalar(0, 255, 255),
	Scalar(255, 255, 0)
};

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
	vector<Point2f> calibPoints;
	vector<int> calibIndex;
	vector<int> pointCluster(width * height);
	bool nightMode = false;
	const int MAX_COUNT = 500;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	Point2f point;
	int hSpread = 20;
	int xBase = 20;
	int vSpread = 20;
	int yBase = 20;
	for (int i = 0; i < height; i++) {
		for (int k = 0; k < width; k++) {
			point = Point2f((float)(k * hSpread) + xBase, (float)(i * vSpread) + yBase);
			points[0].push_back(point);
			pointCluster[(i * width) + k] = 0;
		}
	}
	Mat centers;

	bool calibrate = false;
	auto calibStart = chrono::high_resolution_clock::now();

	while (1) {
		vc.read(frame);
		if (frame.empty()) {
			break;
		}

		//Optical Flow
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if (nightMode) {
			image = Scalar::all(0);
		}
		
		if (listenForKeyPress(VK_SPACE) && !calibrate) {
			cout << "Calibrating..." << endl;
			for (int i = 0; i < pointCluster.size(); i++) {
				pointCluster[i] = 0;
			}
			centers.release();
			calibrate = true;
			calibStart = chrono::high_resolution_clock::now();
		}

		if (!points[0].empty()) {
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty()) {
				gray.copyTo(prevGray);
			}
			
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			float overallMag[NUM_CLUSTERS + 1];
			float vecx[NUM_CLUSTERS + 1], vecy[NUM_CLUSTERS + 1];
			int movePoints[NUM_CLUSTERS + 1];
			float totalMagX[NUM_CLUSTERS + 1];
			float totalMagY[NUM_CLUSTERS + 1];
			for (int index = 0; index < NUM_CLUSTERS + 1; index++) {
				overallMag[index] = 0;
				vecx[index] = 0;
				vecy[index] = 0;
				movePoints[index] = 0;
				totalMagX[index] = 0;
				totalMagY[index] = 0;
			}
			for (i = 0; i < points[0].size(); i++) {
				circle(image, points[0][i], 2, clusterColors[pointCluster[i]], -1, 8);
				if (!status[i])
					continue;

				//points[1][k++] = points[1][i];
				double xdif = points[1][i].x - points[0][i].x;
				double ydif = points[1][i].y - points[0][i].y;
				float magnitude = sqrt(pow(xdif, 2) + pow(ydif, 2));
				if (magnitude >= 5) {
					circle(image, points[0][i], 4, clusterColors[pointCluster[i]], -1, 8);
					arrowedLine(
						image, 
						Point(points[0][i].x, points[0][i].y), 
						Point(points[1][i].x, points[1][i].y),
						Scalar(0, 0, 0),
						1, 
						8, 
						0);
					totalMagX[pointCluster[i]] += xdif;
					totalMagY[pointCluster[i]] += ydif;
					movePoints[pointCluster[i]]++;
					if (calibrate) {
						//Add point to calibrate list
						//Should probably check for duplicates, but this may be a simple way to do weighted variables
						calibPoints.push_back(points[0][i]);
						calibIndex.push_back(i);
					}
				}
			}
			if (countNonZero(centers) < 1) {
				float magx = totalMagX[0] / movePoints[0];
				float magy = totalMagY[0] / movePoints[0];
				overallMag[0] = sqrt(pow(magx, 2) + pow(magy, 2));
				vecx[0] = magx / overallMag[0];
				vecy[0] = magy / overallMag[0];
				if (overallMag[0] > 10 && !calibrate) {
					arrowedLine(
						image,
						Point(image.cols / 2, image.rows / 2),
						Point(vecx[0] * 100 + (image.cols / 2), vecy[0] * 100 + (image.rows / 2)),
						clusterColors[0],
						5,
						8,
						0);
				}
			}
			else {
				for (int j = 1; j < NUM_CLUSTERS + 1; j++) {
					float magx = totalMagX[j] / movePoints[j];
					float magy = totalMagY[j] / movePoints[j];
					overallMag[j] = sqrt(pow(magx, 2) + pow(magy, 2));
					vecx[j] = magx / overallMag[j];
					vecy[j] = magy / overallMag[j];
					if (overallMag[j] > 10 && !calibrate) {
						arrowedLine(
							image,
							Point(centers.at<float>((j - 1) * 2), centers.at<float>((j - 1) * 2 + 1)),
							Point(vecx[j] * 100 + centers.at<float>((j - 1) * 2), vecy[j] * 100 + centers.at<float>((j - 1) * 2 + 1)),
							clusterColors[j],
							5,
							8,
							0);
					}
				}
			}
		}
		else {
			for (int i = 0; i < points[0].size(); i++) {
				circle(image, points[0][i], 3, Scalar(0, 255, 0), -1, 8);
			}
		}

		if (waitKey(10) == 27 && !calibrate) break;

		frameCount++;
		auto end = chrono::high_resolution_clock::now();
		if (chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() > 1000) {
			frameCountKeep = frameCount;
			begin = chrono::high_resolution_clock::now();
			frameCount = 0;
		}
		int countdown = 5 - (int)(chrono::duration_cast<std::chrono::milliseconds>(end - calibStart).count() / 1000);
		if (chrono::duration_cast<std::chrono::milliseconds>(end - calibStart).count() > 5000 && calibrate) {
			cout << "Done calibrating!" << endl;
			Mat labels;
			if (calibPoints.size() > 10) {
				kmeans(calibPoints, NUM_CLUSTERS, labels,
					TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
					3, KMEANS_PP_CENTERS, centers);
				for (int i = 0; i < calibPoints.size(); i++) {
					pointCluster[calibIndex[i]] = labels.at<int>(i) + 1;
				}
			}

			calibPoints.clear();
			calibIndex.clear();
			calibrate = false;
		}
		Mat flip;
		cv::flip(image, flip, 1);
		if (calibrate) {
			cv::putText(
				flip,
				std::to_string(countdown),
				Point(image.cols / 2, image.rows / 2),
				FONT_HERSHEY_COMPLEX_SMALL,
				5,
				Scalar(255, 0, 0),
				1,
				CV_AA
			);
		}
		else {
			cv::putText(
				flip,
				String("FPS:" + std::to_string(frameCountKeep)),
				Point(500, 50),
				FONT_HERSHEY_COMPLEX_SMALL,
				0.8,
				Scalar(0, 0, 0),
				1,
				CV_AA
			);
		}
		cv::imshow("Webcam", flip);
		cv::swap(prevGray, gray);
	}
}