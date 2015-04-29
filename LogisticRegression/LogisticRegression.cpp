#include "LogisticRegression.h"

using namespace std;

/**
 * Logistic regressionの初期化。
 *
 * @param X			入力データ (各行が各入力x_iに相当する）
 * @param Y			出力データ（各行が各出力y_iに相当する）
 * @param lambda	正規化項の係数
 */
LogisticRegression::LogisticRegression(const cv::Mat_<double>& X, const cv::Mat_<double>& Y, float lambda, float alpha, int maxIter) {
	N = X.rows;
	W = cv::Mat_<double>::zeros(X.cols, Y.cols);
	b = cv::Mat_<double>::zeros(1, Y.cols);

	cv::randu(W, -0.1, 0.1);

	train(X, Y, lambda, alpha, maxIter);

	cout << W << endl;
	cout << b << endl;
}

/**
 * 与えられた入力に対する出力を推定する。
 *
 * @param x		入力データ
 * @return		推定された出力データ
 */
cv::Mat_<double> LogisticRegression::predict(const cv::Mat_<double>& X) {
	cv::Mat_<double> tmp;
	cv::exp(-X * W - cv::repeat(b, X.rows, 1), tmp);
	return 1 / (1 + tmp);
}

void LogisticRegression::train(const cv::Mat_<double>& X, const cv::Mat_<double>& Y, float lambda, float alpha, int maxIter) {
	cv::Mat_<double> dW, db;
	for (int iter = 0; iter < maxIter; ++iter) {
		double c = cost(X, Y, lambda);
		cout << "Cost: " << c << endl;

		grad(X, Y, lambda, dW, db);

		W -= dW * alpha;
		b -= db * alpha;
	}
}

double LogisticRegression::cost(const cv::Mat_<double>& X, const cv::Mat_<double>& Y, float lambda) {
	cv::Mat_<double> y_hat = predict(X);

	cv::Mat_<double> log_y_hat, log_one_minus_y_hat;
	cv::log(y_hat, log_y_hat);
	cv::log(1 - y_hat, log_one_minus_y_hat);

	cv::Mat_<double> entropy;
	cv::reduce(Y.mul(log_y_hat) + (1 - Y).mul(log_one_minus_y_hat), entropy, 0, CV_REDUCE_AVG);

	cv::reduce(entropy, entropy, 1, CV_REDUCE_AVG);

	double n = cv::norm(W);
	double cost = -entropy(0, 0) + lambda * n * n;

	return cost;
}

void LogisticRegression::grad(const cv::Mat_<double>& X, const cv::Mat_<double>& Y, float lambda, cv::Mat_<double>& dW, cv::Mat_<double>& db) {
	dW = cv::Mat_<double>::zeros(W.size());
	db = cv::Mat_<double>::zeros(b.size());

	cv::Mat_<double> Y_hat = predict(X);

	for (int r = 0; r < dW.rows; ++r) {
		for (int c = 0; c < dW.cols; ++c) {
			for (int i = 0; i < N; ++i) {
				dW(r, c) -= (Y(i, c) - Y_hat(i, c)) * X(i, r);
				db(r, c) -= Y(i, c) - Y_hat(i, c);
			}
			dW(r, c) = dW(r, c) / N + 2 * lambda * W(r, c);
			db(r, c) = db(r, c) / N;
		}
	}
}

