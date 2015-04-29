#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <random>
#include "LogisticRegression.h"

using namespace std;

double func(double x) {
	return (sin(x * 4.0) + 1.0) * 0.5;
}

int main() {
	std::mt19937 mt(100);
	std::normal_distribution<> normal(0, 1);

	const int N = 100;
	const int NS = 8;
	cv::Mat_<double> X(NS, 1);
	cv::Mat_<double> Y(NS, 1);
	for (int i = 0; i < NS; ++i) {
		double x = (double)i / (double)NS * 2.0 - 1.0;
		X(i, 0) = x;
		Y(i, 0) = func(x);// + normal(mt) * 0.1;
	}

	cout << X << endl;
	cout << Y << endl;

	cv::Mat_<uchar> img = cv::Mat_<uchar>::ones(100, 100) * 255;

	LogisticRegression lr(X, Y, 0.01f, 0.1f, 100);
	for (int i = 0; i < N; ++i) {
		cv::Mat_<double> x(1, 1);
		x(0, 0) = (double)i / (double)N * 2.0 - 1.0;
		cv::Mat_<double> y = lr.predict(x);

		double t = func(x(0, 0));

		//cout << y(0, 0) << endl;

		img((int)((y(0, 0) + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 0;
		img((int)((t + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 128;
	}

	cv::flip(img, img, 0);
	cv::imwrite("test.png", img);

	return 0;
}
