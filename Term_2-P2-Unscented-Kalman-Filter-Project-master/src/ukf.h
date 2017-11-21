#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  //measurement noise covariance matrix
  MatrixXd R_laser_;
  MatrixXd R_radar_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * Augments Sigma Points
   * @param {MatrixXd} Xsig_out The sigma points matrix to be augmented.
   */
  MatrixXd AugmentedSigmaPoints();

  /**
   * Sigma Point Prediction
   * @param {MatrixXd} Xsig_aug The augmented sigma points to be precessed.
   * @param {double} delta_t The time elapsed since last measurement.
   */
  void SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t);

  /**
   * Predict Mean And Covariance
   */
  void PredictMeanAndCovariance();

  /**
   * Predict Laser Measurement
   * @param {MatrixXd} Zsig_out The predicted sigma points in measurement space.
   * @param {VectorXd} z_out The predicted laser state vector.
   * @param {MatrixXd} S_out The predicted laser state covariance matrix.
   */
  void PredictLaserMeasurement(MatrixXd* Zsig_out, VectorXd* z_out, MatrixXd* S_out);

  /**
   * Predict Radar Measurement
   * @param {MatrixXd} Zsig_out The predicted sigma points in measurement space.
   * @param {VectorXd} z_out The predicted radar state vector.
   * @param {MatrixXd} S_out The predicted radar state covariance matrix.
   */
  void PredictRadarMeasurement(MatrixXd* Zsig_out, VectorXd* z_out, MatrixXd* S_out);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates state the state, and the state covariance matrix.
   * @param {int} n_z measurement dimension
   * @param {MatrixXd} Zsig sigma points in measurement space
   * @param {VectorXd} z_pred vector for mean predicted measurement
   * @param {MatrixXd} S matrix for predicted measurement covariance
   * @param {VectorXd} z vector for incoming measurement
   */
  void UpdateState(int n_z, MatrixXd Zsig, VectorXd z_pred, MatrixXd S, VectorXd z);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
