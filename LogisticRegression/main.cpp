#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <random>
#include "LogisticRegression.h"

using namespace std;

cv::Mat_<double> func(cv::Mat_<double> x) {
	cv::Mat_<double> ret(1, 2);
	ret(0, 0) = sqrt(x(0, 0) * x(0, 0) + x(0, 1) * x(0, 1));
	ret(0, 1) = x(0, 0) * x(0, 1);
	return ret;
}

int main() {
	std::mt19937 mt(100);
	std::normal_distribution<> normal(0, 1);

	const int N = 100;
	const int NS = 1;
	cv::Mat_<double> X(NS, 1);
	cv::Mat_<double> Y(NS, 2);
	for (int i = 0; i < NS; ++i) {
		double x1 = normal(mt);
		double x2 = normal(mt);
		X(i, 0) = x1;
		//X(i, 1) = x2;
		cv::Mat_<double> x = (cv::Mat_<double>(1, 2) << x1, x2);
		cv::Mat_<double> y = func(x);
		Y(i, 0) = y(0, 0);
		Y(i, 1) = y(0, 1);
	}

	cout << X << endl;
	cout << Y << endl;

	//cv::Mat_<uchar> img = cv::Mat_<uchar>::ones(100, 100) * 255;

	LogisticRegression lr(X, Y, 0.01f, 0.1f, 100);
	for (int i = 0; i < N; ++i) {
		double x1 = normal(mt);
		double x2 = normal(mt);
		cv::Mat_<double> x = (cv::Mat_<double>(1, 2) << x1, x2);
		cv::Mat_<double> y = lr.predict(x);

		cv::Mat_<double> t = func(x);

		//cout << y(0, 0) << endl;

		//img((int)((y(0, 0) + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 0;
		//img((int)((t + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 128;
	}

	/*
	cv::flip(img, img, 0);
	cv::imwrite("test.png", img);
	*/

	return 0;
}
