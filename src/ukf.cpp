#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ =  7;//30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;//30

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;
    
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;  

  x_aug_ = VectorXd(n_aug_);
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);

  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;


}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {		
    if (is_initialized_ == false) {
		float px;
		float py;
		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			px = meas_package.raw_measurements_(0);
			py = meas_package.raw_measurements_(1);
		}
		else {
			px = meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
			py = meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
		}		
		x_ << px,//5.7441,
			py,//1.3800,
			0, 0, 0;
			//2.2049,
			//0.5015,
			//0.3528; //px, py, 0, 0, 0;
				
		P_ << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
			-0.0013, 0.0077, 0.0011, 0.0071, 0.0060,
			0.0030, 0.0011, 0.0054, 0.0007, 0.0008,
			-0.0022, 0.0071, 0.0007, 0.0098, 0.0100,
			-0.0020, 0.0060, 0.0008, 0.0100, 0.0123; 

		
		generateSigmaPoints();

		time_us_ = meas_package.timestamp_;

		is_initialized_ = true;		
		return;
	}
	std::cout << x_ << std::endl;

	
	double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;
		
	generateSigmaPoints();
	Prediction(delta_t);

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {		
		UpdateRadar(meas_package);
	}
	else {
		UpdateLidar(meas_package);
	}
	

}


void UKF::generateSigmaPoints() {
	//create augmented mean state
	x_aug_.head(n_x_) = x_;
	x_aug_(n_x_) = 0;
	x_aug_(n_x_ + 1) = 0;

	//create augmented covariance matrix
	P_aug_.fill(0.0);
	P_aug_.topLeftCorner(n_x_, n_x_) = P_;
	P_aug_(n_x_, n_x_) = std_a_ * std_a_;
	P_aug_(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
	P_aug_(n_x_, n_x_ + 1) = 0;
	P_aug_(n_x_ + 1, n_x_) = 0;

	//create square root matrix
	MatrixXd P_aug_sqrt = P_aug_.llt().matrixL();

	//create augmented sigma points	
	float cons = sqrt(lambda_ + n_aug_);
	Xsig_aug_.col(0) = x_aug_;
	for (int i = 0; i < n_aug_; ++i)
	{
		Xsig_aug_.col(i + 1) = x_aug_ + cons * P_aug_sqrt.col(i);
		Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - cons * P_aug_sqrt.col(i);
	}
}
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  
	MatrixXd x = MatrixXd(n_aug_, 1);
	MatrixXd x_t = MatrixXd(n_x_, 1);
	MatrixXd tmp = MatrixXd(n_x_, 1);
	MatrixXd noise = MatrixXd(n_x_, 1);
	//predict sigma points
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		x = Xsig_aug_.col(i);
		x_t = Xsig_aug_.col(i).head(n_x_);

		//avoid division by zero
		if (x(4) > 0.001) {
			tmp << x(2) / x(4) * (sin(x(3) + x(4)*delta_t) - sin(x(3))),
				x(2) / x(4) * (-cos(x(3) + x(4)*delta_t) + cos(x(3))),
				0,
				x(4) * delta_t,
				0;
		}
		else {
			tmp << x(2) * cos(x(3)) * delta_t,
				x(2) * sin(x(3)) * delta_t,
				0,
				x(4) * delta_t,
				0;
		}

		noise << 0.5 * (delta_t * delta_t) * cos(x(3)) * x(5),
			0.5 * (delta_t * delta_t) * sin(x(3)) * x(5),
			delta_t * x(5),
			0.5 * (delta_t * delta_t) *  x(6),
			delta_t *  x(6);
		
		//write predicted sigma points into right column
		Xsig_pred_.col(i) = x_t + tmp + noise;
		
	}
	
	PredictMeanAndCovariance(x_, P_);
	
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //create matrix for sigma points in measurement space

	int n_z = 2;
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	MatrixXd tmp = MatrixXd(n_z, 1);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);		
	
	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {		
		Zsig.col(i) << Xsig_pred_.col(i)(0), Xsig_pred_.col(i)(1);
	}

	
	//calculate mean predicted measurement
	z_pred.fill(0.0);	
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}
	//std::cout << "z_pred" << z_pred << std::endl;
	//std::cout << "Xsig_pred_" << Xsig_pred_ << std::endl;

	

	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		tmp = Zsig.col(i) - z_pred;
		S = S + weights_(i) * tmp * tmp.transpose();
	}
	S = S + R;

	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);
	//calculate cross correlation matrix
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd x_diff = (Xsig_pred_.col(i) - x_);
		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		Tc = Tc + weights_(i) * x_diff * (Zsig.col(i) - z_pred).transpose();
	}
	//calculate Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//update state mean and covariance matrix
	x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
	P_ = P_ - K * S * K.transpose();

	NIS_laser_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() * (meas_package.raw_measurements_ - z_pred);

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //create matrix for sigma points in measurement space
	int n_z = 3;
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	MatrixXd tmp = MatrixXd(n_z, 1);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		tmp = Xsig_pred_.col(i);
		float rho = sqrt(tmp(0) * tmp(0) + tmp(1) * tmp(1));
		float phi = atan2(tmp(1), tmp(0));
		float rho_d = (tmp(0)*cos(tmp(3)) * tmp(2) + tmp(1)*sin(tmp(3))*tmp(2)) / rho;
		Zsig.col(i) << rho, phi, rho_d;
	}
	//calculate mean predicted measurement
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}
	//calculate measurement covariance matrix S
	MatrixXd R = MatrixXd(n_z, n_z);
	S.fill(0.0);
	R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_ * std_radrd_;
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		tmp = Zsig.col(i) - z_pred;
		S = S + weights_(i) * tmp * tmp.transpose();
	}
	S = S + R;	

	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);
	//calculate cross correlation matrix
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd x_diff = (Xsig_pred_.col(i) - x_);
		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		VectorXd z_diff = (Zsig.col(i) - z_pred);
	    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	//calculate Kalman gain K;
	MatrixXd K = Tc * S.inverse();	

	//update state mean and covariance matrix
	VectorXd z_diff = (meas_package.raw_measurements_ - z_pred);
	while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
	while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();
	NIS_radar_= (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() * (meas_package.raw_measurements_ - z_pred);
}


void UKF::PredictMeanAndCovariance(VectorXd& x, MatrixXd& P) {

	MatrixXd tmp = MatrixXd(n_x_, 1);
	P.fill(0.0);
	
	//set weights
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i < weights_.size(); ++i) {
		weights_(i) = 0.5 * 1 / (lambda_ + n_aug_);
	}

	//predict state mean
	x.fill(0.0);	
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		x = x + weights_(i) * Xsig_pred_.col(i);
	}
	//predict state covariance matrix

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		tmp = Xsig_pred_.col(i) - x;
		while (tmp(3) > M_PI) tmp(3) -= 2.*M_PI;
	    while (tmp(3) < -M_PI)tmp(3) += 2.*M_PI;
		P = P + weights_(i) * tmp * tmp.transpose();
	}
}


