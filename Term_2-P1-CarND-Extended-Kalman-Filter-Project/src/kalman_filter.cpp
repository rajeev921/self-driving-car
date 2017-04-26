#include "kalman_filter.h"

#define PI 3.14159265

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd radar_Prediction(const VectorXd &x_state);

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */  
   
  VectorXd z_pred = H_ * x_;
//  z_pred = atan2(z_pred, -z_pred) * 180 / PI;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_; 
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt *Si;

  // new estimate 
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

  VectorXd z_pred = radar_Prediction(x_);
//z_pred = atan2(z_pred, -z_pred) * 180 / PI;
  VectorXd y = z - z_pred;

   while(y[1] > PI || y[1] < -PI)
   {
	if(y[1] > PI)
       		 y[1]-=PI;
        else y[1]+=PI;
   }

  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}

VectorXd radar_Prediction(const VectorXd &x_state)
{
   VectorXd z_radar(3);
   double rho, phi, rho_dot;

   // recover state parameter
   double px = x_state(0);
   double py = x_state(1);
   double vx = x_state(2);
   double vy = x_state(3);

   rho = sqrt(px*px +py*py);

   phi = atan2(py ,px);

   if(fabs(rho) < .0001) {
        rho_dot = (px*vx + py*vy) / 0.0001;
   }
   else {
        rho_dot = (px*vx + py*vy) / rho;
   }

   z_radar << rho, phi, rho_dot;

   return z_radar;
}


//hx[2] = (px*vx + py*vy)/max(.0001,hx[0]);
