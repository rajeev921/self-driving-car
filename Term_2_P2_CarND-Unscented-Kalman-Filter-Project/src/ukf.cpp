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
  //initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

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

  // state dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  //create vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  // The current NIS for radar
  //  NIS_radar;

 // The current NIS for laser
  //  NIS_laser ;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if(!is_initialized_)
  {  
  	/* Initialized the ukf.x with the first measurement */	
	// first measurement
	cout << "UKF: " << endl; 

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        /**
         Convert radar from polar to cartesian coordinates and initialize state.
        */

            float ro = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float rho_dot = measurement_pack.raw_measurements_[2];

            x_[0] = ro * cos(phi);
            x_[1] = ro * sin(phi);
            x_[2] = rho_dot;
            x_[3] =  phi;
            x_[4] =  0;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */

      x_[0] = measurement_pack.raw_measurements_[0];
      x_[1]  = measurement_pack.raw_measurements_[1];
      x_[2] =  0;
      x_[3] =  0;
      x_[4] =  0;

   }
    // done initializing, no need to predict or update
    is_initialized_ = true;

    // Initialize timestamp
    time_us_ = measurement_pack.timestamp_;
	
  }
  
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

    // calculate the delta_t 
    double delta_t = (measurement_pack.timestamp_ - time_us_) / 1000000.0;
    // update time
    time_us_ = measurement_pack.timestamp_;    

    Prediction(delta_t);

    /*****************************************************************************
    *  Prediction
    ****************************************************************************/

   if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
   {
	UpdateRadar(measurement_pack);      
      
   }
   else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
   {
	UpdateLidar(measurement_pack);

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

   // 1. Generate sigma points 
   
   // create augmented mean vector
   VectorXd x_aug_ = VectorXd(7);
   
   x_aug_.head(5) = x_;
   x_aug_(5) = 0;
   x_aug_(6) = 0;

   // Created augmented state covariance
   MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);

   P_aug_.fill(0.0);
   P_aug_.topLeftCorner(5, 5) = P_;
   P_aug_(5, 5) = std_a_ * std_a_;
   P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

   //calculate square root of P
   MatrixXd A = P_.llt().matrixL();

   // create sigma points matrix
   MatrixXd Xsig_aug_;
   Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

   //std::cout << "Root of covariance Matrix = " << std::endl << A << std::endl;

   // set first column of augmented sigma point matrix
   Xsig_aug_.col(0) = x_aug_;

   // set remaining sigma points
   for(int i = 0; i <n_x_; i++)
   {
        Xsig_aug_.col(i+1)     = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i);

        Xsig_aug_.col(i+1+n_x_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i);
   }


   // 2. Predict sigma points
   Xsig_pred_ =  MatrixXd(n_x_, 2 * n_aug_ + 1);
   
   for(int i = 0; i < 2 * n_aug_ + 1; i++)
   {
	double p_x      = Xsig_aug_(0, i);
	double p_y      = Xsig_aug_(1, i);
	double v        = Xsig_aug_(2, i); 
	double yaw      = Xsig_aug_(3, i);
	double yawd     = Xsig_aug_(4, i);
	double nu_a     = Xsig_aug_(5, i);
	double nu_yawdd = Xsig_aug_(6, i);

	// predicted state value
        double px_p, py_p;

	// avoid division by zero
	if (fabs(yawd) > 0.001) { 
	    px_p = p_x + v/yawd * ( sin(yaw + yawd*delta_t) - sin(yaw) );
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
        }
	else { 
	    px_p = p_x + v * delta_t * cos(yaw);
	    py_p = p_y + v * delta_t * sin(yaw);
	}

	double v_p    = v;
	double yaw_p  = yaw + yawd * delta_t;
	double yawd_p = yawd;

	// Add noise
	px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
	py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
	v_p  = v_p + nu_a * delta_t; 

	yaw_p  = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
	yawd_p = yawd_p + nu_yawdd * delta_t;
 
        Xsig_pred_(0, i) = px_p;
	Xsig_pred_(1, i) = py_p;
	Xsig_pred_(2, i) = v_p;
	Xsig_pred_(3, i) = yaw_p;
	Xsig_pred_(4, i) = yawd_p;
   } 
  
   // 3. Predict mean and covariance

   // Set weights
   double weight_0 = lambda_ / (lambda_ + n_aug_);
   weights_(0) = weight_0;
   for(int i = 0; i < 2 * n_aug_ + 1; i++)
   {
	double weight = 0.5 * lambda_ / (lambda_ + n_aug_);
	weights_(i) = weight;
   }

   // predicted state mean Vector 
   for(int i = 0; i < 2 * n_aug_ + 1; i++)
   {
	x_ += weights_(i) *Xsig_pred_.col(i);
   }

   // predicted state covariance matrix
  
   for(int i = 0; i < 2 * n_aug_ + 1; i++)
   {
	VectorXd x_diff = Xsig_pred_.col(i) - x_;

	// Angle Normalization
	while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
	while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

	P_ += weights_(i) * x_diff * x_diff.transpose();
   }
   
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

  //set measurement dimension, radar can measure px and py
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i  = 0; i < 2 * n_aug_ + 1; i++) {  // 2n+1 sigma points

	double p_x  = Xsig_pred_(0, i);
	double p_y  = Xsig_pred_(1, i);

	Zsig(0, i) = p_x;
	Zsig(1, i) = p_y;
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2*n_aug_; i++)  {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)  { //2n+1 sigma points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<  std_laspx_*std_laspx_,
        std_laspy_*std_laspy_;

  S = S + R;

  // Incoming laser measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {   //2n+1 sigma points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalization
        while(z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
        while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        //state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // angle normalization
        while(x_diff(1) >  M_PI) x_diff(1) -= 2. * M_PI;
        while(x_diff(1) < -M_PI) x_diff(1) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff *z_diff.transpose();
  }

  //kalman gain k
  MatrixXd k = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // agle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + k * z_diff;
  P_ = P_ + k * S * k.transpose(); 	  

  NIS_laser_ = (z - z_pred).transpose() * S *(z - z_pred);
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

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i  = 0; i < 2 * n_aug_ + 1; i++) {  // 2n+1 sigma points

        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v   = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

	double v1 = cos(yaw)*v;
	double v2 = sin(yaw)*v;

	// measurement model
	Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y); 			    // r
	Zsig(1, i) = atan2(p_y, p_x);				    // phi
	Zsig(2, i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);  // r_dot
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2*n_aug_; i++)  {
	z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)  { //2n+1 sigma points
	// residual
	VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_ * std_radr_, 0, 0,
          0, std_radphi_ * std_radphi_, 0,
          0, 0,std_radrd_ * std_radrd_;
  S = S + R;  

  //incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {   //2n+1 sigma points

	//residual
	VectorXd z_diff = Zsig.col(i) - z_pred;

	// angle normalization
	while(z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
	while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

	//state difference
	VectorXd x_diff = Xsig_pred_.col(i) - x_;

	// angle normalization
	while(x_diff(1) >  M_PI) x_diff(1) -= 2. * M_PI;
        while(x_diff(1) < -M_PI) x_diff(1) += 2. * M_PI;

	Tc = Tc + weights_(i) * x_diff *z_diff.transpose();
  }

  //Kalman gain k
  MatrixXd k = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // agle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + k * z_diff;
  P_ = P_ + k * S * k.transpose();

  NIS_radar_ = (z - z_pred).transpose() * S * (z - z_pred);
  
}

