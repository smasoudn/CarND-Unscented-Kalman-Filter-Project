#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;
Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;
	if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
		std::cout << "Estimations and groud truth vectors must have the same size." << std::endl;
		return rmse;
	}


	for (int i = 0; i < estimations.size(); ++i) {

		VectorXd diff = (estimations[i] - ground_truth[i]);
		VectorXd diff2 = diff.array() * diff.array();
		rmse = rmse + diff2;
	}
	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();

	return rmse;
}
