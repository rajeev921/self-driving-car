/*
 * particle_filter.cpp
 * 
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */

	num_particles = 100; // 1000

	default_random_engine gen;

	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);


	for(int i = 0; i < num_particles; ++i) 
	{
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1);

	}
	
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */

	default_random_engine gen;
	
	for(int i = 0; i < num_particles; i++)
	{

		double new_x;
		double new_y;
		double new_theta;

		if (yaw_rate == 0)  // fabs(yaw_rate) > 0.001
		{
			new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else{
			new_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate*delta_t;
		}	

		//Adding the jitter

		normal_distribution<double> N_x(new_x, std_pos[0]);
		normal_distribution<double> N_y(new_y, std_pos[1]);
		normal_distribution<double> N_theta(new_theta, std_pos[2]);

		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
		
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */

	for (int i = 0; i < observations.size(); i++) {
		LandmarkObs observation = observations[i];
		double minimum_distance = INFINITY;
		observations[i].id = -1;

		for (int j = 0; j < predicted.size(); j++) {
			LandmarkObs prediction = predicted[j];
			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);

			if (distance < minimum_distance) {
				minimum_distance = distance;
				observations[i].id = prediction.id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */

	double cov_x = std_landmark[0] * std_landmark[0];
	double cov_y = std_landmark[1] * std_landmark[1];
	double normalizer = 2.0 * M_PI * std_landmark[0] * std_landmark[1];

	for (int i = 0; i < particles.size(); i++) {
		vector<LandmarkObs> predicted_landmarks;
		for (auto map_lm : map_landmarks.landmark_list) {
			LandmarkObs pred_lm;
			pred_lm.x = map_lm.x_f;
			pred_lm.y = map_lm.y_f;
			pred_lm.id = map_lm.id_i;

			if (dist(pred_lm.x, pred_lm.y, particles[i].x, particles[i].y) <= sensor_range) {
				predicted_landmarks.push_back(pred_lm);
			}
		}

		vector<LandmarkObs> transformed_obs;
		for (auto obs_lm : observations) {
			LandmarkObs obs_global;
			obs_global.id = obs_lm.id;
			obs_global.x = obs_lm.x * cos(particles[i].theta) - obs_lm.y * sin(particles[i].theta) + particles[i].x;
			obs_global.y = obs_lm.x * sin(particles[i].theta) + obs_lm.y * cos(particles[i].theta) + particles[i].y;
			transformed_obs.push_back(obs_global);
		}

		dataAssociation(predicted_landmarks,transformed_obs);
		particles[i].weight = 1.0;

		for (int j = 0; j < transformed_obs.size(); j++) {
			double pred_x, pred_y, obs_x, obs_y;
			auto transformed_ob = transformed_obs[j];

			for (int k = 0; k < predicted_landmarks.size(); k++) {
				if (predicted_landmarks[k].id == transformed_ob.id) {
					pred_x = predicted_landmarks[k].x;
					pred_y = predicted_landmarks[k].y;
				}
			}

			double dist_x = (pred_x - transformed_obs[j].x);
			double dist_y = (pred_y - transformed_obs[j].y);
			double pdf = exp(-(dist_x * dist_x/(2*cov_x) + dist_y* dist_y/(2*cov_y))) / normalizer;

			particles[i].weight *= pdf;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */

    /*
	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	std::vector<Particle> resample_particles;

	for ( int i = 0; i < num_particles; i++)
	{

		resample_particles.push_back(particles[distribution(gen)]);

	}

	particles = resample_particles;

	*/

	default_random_engine generator;
	vector<Particle> new_particles;

	vector<double> weights;
	for (int i = 0; i < particles.size(); i++) {
		weights.push_back(particles[i].weight);
	}

	double beta = 0.0;
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_int_distribution<int> int_distribution(0, num_particles - 1);
	uniform_real_distribution<double> real_distribution(0.0, max_weight);

	auto index = int_distribution(generator);
	for (int i = 0; i < particles.size(); i++) {
		beta += real_distribution(generator) * 2.0;

		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}

		new_particles.push_back(particles[index]);
	}

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
