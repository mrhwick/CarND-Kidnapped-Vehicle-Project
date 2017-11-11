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
	// TODO: Set the number of particles. Initialize all particles to first 
	// position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about 
	//  this method (and others in this file).

	num_particles = 50;
	std::default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		
		particles.push_back(p);
		weights.push_back(1.0);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	double yawdt = yaw_rate * delta_t;
	double velyaw = velocity / yaw_rate;
	double veldt = velocity * delta_t;

	for (int i = 0; i < num_particles; i++) {
		Particle &p = particles[i];

		double cos_theta = cos(p.theta);
		double sin_theta = sin(p.theta);
		
		if (fabs(yaw_rate) <= 0.0001) {
			p.x += veldt * cos_theta + dist_x(gen);
			p.y += veldt * sin_theta + dist_y(gen);
		} else {
			p.x += (velyaw * (sin(p.theta + yawdt) - sin_theta)) + dist_x(gen);
			p.y += (velyaw * (cos_theta - cos(p.theta + yawdt))) + dist_y(gen);
		}

		p.theta += yawdt + dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to 
	// each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. 
	// But you will probably find it useful to 
	//   implement this method and use it as a helper during 
	// the updateWeights phase.

	// cout << "observations.size(): " << observations.size() << endl;
	for (int i = 0; i < observations.size(); i++) {
		// cout << "on observation " << i << endl;
		LandmarkObs &o = observations.at(i);
		o.id = -1;
		double min_value = MAXFLOAT;
		// cout << "predicted.size(): " << predicted.size() << endl;
		for (int j = 0; j < predicted.size(); j++) {
			// cout << "on prediction " << j << endl;
			LandmarkObs plm = predicted.at(j);
			double distance = dist(o.x, o.y, plm.x, plm.y);
			// cout << distance << endl;
			if (distance < min_value) {
				// cout << "Found a minimum" << endl;
				min_value = distance;
				o.id = j;
			}
		}
		// cout << observation.id << endl;
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
	//   http://planning.cs.uiuc.edu/node99.html

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double std_x_sq_2 = 2.0 * std_x * std_x;
	double std_y_sq_2 = 2.0 * std_y * std_y;
	double normalization = 1.0 / (2.0 * M_PI * std_x * std_y);

	for (int i = 0; i < num_particles; i++) {
		Particle &p = particles[i];
		vector<LandmarkObs> landmark_candidates;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s &lm = map_landmarks.landmark_list[j];
			double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
			if (distance <= sensor_range) {
				LandmarkObs lm_candidate;
				lm_candidate.id = lm.id_i;
				lm_candidate.x = lm.x_f;
				lm_candidate.y = lm.y_f;

				landmark_candidates.push_back(lm_candidate);
			}
		}

		vector<LandmarkObs> map_observations;
		double cos_theta = cos(p.theta);
		double sin_theta = sin(p.theta);
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs o = observations[j];

			double ox = p.x + cos_theta * o.x + (- sin_theta * o.y);
			double oy = p.y + sin_theta * o.x + cos_theta * o.y;
			o.x = ox;
			o.y = oy;
			map_observations.push_back(o);
		}

		// cout << "Beginning dataAssociation()" << endl;
		dataAssociation(landmark_candidates, map_observations);
		// cout << "Finished dataAssociation()" << endl;

		p.weight = 1.0;
		for (int j = 0; j < map_observations.size(); j++) {
			LandmarkObs o = map_observations.at(j);
			if (o.id > 0) {

				// cout << observation.id << " out of " << landmark_candidates.size() << endl;
				LandmarkObs &nearest = landmark_candidates.at(o.id);

				double x_diff = o.x - nearest.x;
				double y_diff = o.y - nearest.y;
				double x_diff_sq = x_diff * x_diff;
				double y_diff_sq = y_diff * y_diff;

				double exponent = (x_diff_sq / std_x_sq_2) + 
									(y_diff_sq / std_y_sq_2);
				exponent = exp(-exponent);
				double weight_add = normalization * exponent;

				// cout << weight_add << endl;
				p.weight *= weight_add;
			}
		}
		weights[i] = p.weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine gen;
	discrete_distribution<> dist(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for (int i = 0; i < num_particles; i++) {
		new_particles.push_back(particles[dist(gen)]);
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
