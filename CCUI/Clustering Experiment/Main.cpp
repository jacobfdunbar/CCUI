#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <Windows.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

const int POINTS_SIZE = 1000;
const int MAX_CLUSTERS = 5;
Scalar clusterColors[] =
{
	Scalar(0, 0, 255),
	Scalar(0, 255, 0),
	Scalar(255, 0, 0),
	Scalar(255, 255, 0),
	Scalar(0, 255, 255)
};

int main(int argc, char ** argv) {
	while (1) {
		cout << "Clustering Test" << endl;
		vector<Point2f> points(POINTS_SIZE);
		for (int i = 0; i < POINTS_SIZE; i++) {
			int randx = rand() % 500 + 1;
			int randy = rand() % 500 + 1;
			points[i] = Point2f(randx, randy);
		}
		Mat img_orig(500, 500, CV_8UC3), img_new(500, 500, CV_8UC3);
		Mat labels, centers;
		auto begin = chrono::high_resolution_clock::now();
		kmeans(points, 2, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);
		auto end = chrono::high_resolution_clock::now();
		cout << "Time taken for kmeans: " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << endl;
		img_orig = Scalar::all(0);
		img_new = Scalar::all(0);
		for (int i = 0; i < POINTS_SIZE; i++) {
			int clusterIndex = labels.at<int>(i);
			Point2f pnt = points.at(i);
			circle(img_orig, pnt, 2, Scalar(255, 255, 255), FILLED, LINE_AA);
			circle(img_new, pnt, 2, clusterColors[clusterIndex], FILLED, LINE_AA);
		}

		imshow("Original", img_orig);
		imshow("Clusters", img_new);

		while (1) {
			if (waitKey(10) == 27) break;
		}
	}
}