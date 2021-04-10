import random
import math
import numpy as np
import copy

from math import acos, sin, cos
from cmath import exp

import sys, os, time

from qinfer import LiuWestResampler
from qinfer import utils

from qinfer import FiniteOutcomeModel, DifferentiableModel
from qinfer.smc import SMCUpdater


class Distribution():
    
    def __init__(self, distribution_generator=None,
                 n_points=100,
                 n_qubits=1, 
                 rejection_threshold=0.5,
                 resampler_a=np.sqrt(0.9),
                 resampler_h=np.sqrt(0.1),
                 percentage_tested=0.01,
                 points_to_sample=None):
        
        
        self.vals_per_qubit = 2
        if distribution_generator is not None:
            self.points, self.weights = distribution_generator(n_points, n_qubits)
        else:
            self.points, self.weights = self.uniform_distribution(n_points=n_points, n_qubits=n_qubits)
        
        self.resampler_a = resampler_a
        self.resampler_h = resampler_h
        
        self.n_qubits = n_qubits
        self.n_points = len(self.weights)
        self.rejection_threshold = rejection_threshold
        self.covariance_matrix = None
        
        self.curr_measurement = None
        self.prev_measurement = None
        
        if points_to_sample is None:
            self.points_to_sample = max(1, int(percentage_tested * n_points))
        else:
            self.points_to_sample = points_to_sample

        self.splitting_threshold = 0.1
        
        
    def measure(self, measurement_data, measurement_target, inversion_arr=None, resample=True):
        
        if resample:
            self.conditional_resample()
                
        # Ensure the measurement data is a numpy array
        measurement_data = np.array(measurement_data)
        
        new_weights = []
        for i in range(self.n_points):
            new_weights.append(self.update_estimate(
                self.points[i], 
                self.weights[i], 
                measurement_target, 
                measurement_data))

        self.weights = np.array(new_weights)
        self.renormalise()
                      
        
        return

    def renormalise(self):
        total = sum(self.weights)
        self.weights = np.array([i / total for i in self.weights])

        
        
    def calc_covariance_matrix(self):
        '''
            Calculates the covariance matrix
        '''
        return utils.particle_covariance_mtx(self.weights, self.points)     
        
    
    def resample(self):
        resampler = LiuWestResampler()
        self.points, self.weights = resampler(self.points, self.weights)
        

    # def great_circle_distance(self, point_a, point_b):
    #     return acos(sin(point_a[0]) * sin(point_b[0]) + cos(point_a[1]) * cos(point_b[1]) * cos(point_b[1] - point_a[1]))
        
    def rotation_matricies(self, point):
        theta, phi = point
        r_x = np.array([[cos(theta / 2), -1j * sin(theta / 2)],[-1j * sin(theta / 2), cos(theta / 2)]])
        r_z = np.array([[exp(1j * phi / 2), 0j], [0j, exp(-1j * phi / 2)]])
        return r_x, r_z

    # Replace this bias with the covariance
    def update_estimate(self, point, weight, applied_unitary, measurement_outcome):
        '''
            Calculate the new weight from the previous one
        '''
        total_weight = weight
                
        for i, measurement_result in zip(range(self.n_qubits), measurement_outcome):
            # Points are in groups of three
            values = np.array(point[i * 2 : i * 2 + 2])

            initial_state = np.array([[1], [0]])
            
            r_x_est, r_z_est = self.rotation_matricies(point)
            r_x_app, r_z_app = self.rotation_matricies(applied_unitary)

            s_est = r_x_est @ r_z_est @ initial_state
            s_app = r_x_app @ r_z_app @ initial_state

            p_correct = (np.abs(s_app.T @ s_est) ** 2)[0,0]

            expected_result = np.round(p_correct)

            if measurement_result == expected_result:
                total_weight *= p_correct / self.n_qubits / pow(self.n_qubits, 0.5)
            else:
                total_weight *= (1 - p_correct) / self.n_qubits / pow(self.n_qubits, 0.5)
                
        return total_weight

    def calc_bayes_mean(self):
        bayes_mean = 0

        for point, weight in zip(self.points, self.weights):
            bayes_mean += point * weight
        return bayes_mean
            
        
    def random_distribution(self, n_points, start=0, stop=2 * np.pi, n_qubits=1):
        
        points = []
        
        for _ in range(n_points):
            points.append([np.random.uniform(- np.pi), np.random.uniform(np.pi)])

            
        weights = np.array([1 / len(points)] * len(points))
        points = np.array(points)
        
        return points, weights
        

    def uniform_distribution(self, n_points, start=0, stop=2 * np.pi, n_qubits=1):
        
        n_points_side = int(pow(n_points, 0.5))
        points = []

        for x in np.linspace(-np.pi, np.pi, n_points_side):
            for y in np.linspace(-np.pi, np.pi, n_points_side):
                points.append([x, y])
            
        weights = np.array([1 / len(points)] * len(points))
        points = np.array(points)
        
        return points, weights
    
    def calc_best_experiment(self, acc=0.1, count=None):
        
        # Avoiding problems
        # Being lazy, etc
        if count == None:
            count = 2 * self.n_qubits
        
        if count == 0:
            return 
        
        mean = self.calc_bayes_mean()
        bounds = self.calc_bayes_risk() * acc
        
        highest_points = list(zip(self.weights, self.points))
        highest_points.sort(key=lambda x: x[0], reverse=True)
        highest_points = highest_points[:int(len(highest_points) * acc)]
        
        
        for i, val in enumerate(mean):
            in_range = False
            
            while not in_range:
                for j in highest_points:
                    if abs(val - j[1][i]) < bounds:
                        in_range = True
                        break
                        
                if not in_range:
                    
                    # Split distribution
                    new_points = [[], []]
                    new_weights = [[], []]

                    for j in zip(self.weights, self.points):
                        if j[1][i] <= val:
                            new_weights[0].append(copy.deepcopy(j[0]))
                            new_points[0].append(copy.deepcopy(j[1]))
                        else:
                            new_weights[1].append(copy.deepcopy(j[0]))
                            new_points[1].append(copy.deepcopy(j[1]))
                                 
                    if sum(new_weights[0]) > sum(new_weights[1]):
                        target = 0
                    else:
                        target = 1
                    
                    new_points = new_points[target]
                    new_weights = new_weights[target]
                    
                    sub_dist = Distribution(n_points=len(new_weights), n_qubits=self.n_qubits)
                    
                    sub_dist.points = new_points
                    sub_dist.weights = new_weights
                    sub_dist.renormalise()
                    
                    return sub_dist.calc_best_experiment(count=count - 1)
            
        return mean


    def calc_bayes_risk(self):
        '''
        Simple model using mean squared error as the loss function
        This sets the estimate to the bayes mean of the posterior
        '''
        bayes_risk = 0
        estimate = self.calc_bayes_mean()
        for point, weight in zip(self.points, self.weights):
            bayes_risk += weight * sum((point - estimate) ** 2)
        return bayes_risk
        
    
    def n_eff(self):
        return 1 / sum(i ** 2 for i in self.weights)
    
    def conditional_resample(self):
        
        # Calculate number of effective particles        
        if self.n_eff() < self.n_points * self.rejection_threshold:
            #print("RESAMPLE")
            self.resample()
    
    
    def resample(self):
        # Adapted from qinfer resamplers.py
        bayes_mean = self.calc_bayes_mean()
        covariance = self.calc_covariance_matrix()
        
        # Calculate square root psd
        square_root_psd, square_root_psd_err = utils.sqrtm_psd(covariance)
        square_root_psd = np.real(self.resampler_h * square_root_psd)
        
        # Create new points and related indicies
        new_points = np.empty(self.points.shape)
        cumulative_sum_weights = np.cumsum(self.weights)
        particles_to_resample = np.arange(self.n_points, dtype=int)
        
        # Draw from particles with probability dictated by weights
        redrawn_indicies = cumulative_sum_weights.searchsorted(
            np.random.random((particles_to_resample.size,)),
            side='left'
        )
        
        # Draw offsets
        offsets = self.resampler_a * self.points[redrawn_indicies,:] 
        + (1 - self.resampler_a) * bayes_mean
        
        # Going to skip boundary condition checks for this one...
        new_points[particles_to_resample] = offsets + np.dot(
            square_root_psd, np.random.randn(self.n_qubits * self.vals_per_qubit, offsets.shape[0])).T

        resampled_locations = new_points[particles_to_resample, :]
        
        
        # Reset the density to be uniform, the weight information is 
        # now conveyed within the distribution of the particles
        self.points = resampled_locations
        
        # Very very simple boundary enforcement
        for i, point in enumerate(self.points):
            for j, val in enumerate(point):
                self.points[i][j] %= 2 * np.pi
                
        self.weights = np.array([1 / self.n_points] * self.n_points)
   
    def curr_best_circuit(self):
        bayes_mean = self.calc_bayes_mean()
        
        
        
    def linear_distribution(self, step, start=0, stop=1, n_qubits=1):
        '''
            Simple linear, evenly weighted distribution
        '''
        points = []
        
        position = start
        while position <= stop:
            points.append(position)
            position += step
        
        if stop > points[-1]:
            points.append(stop)
            
        weights = np.array([1 / len(points)] * len(points))
        
        points = np.array(points)
        
        return points, weights

    def bloch_distance(self, point_a, point_b):
        initial_state = np.array([[1], [0]])
            
        r_x_est, r_z_est = self.rotation_matricies(point_a)
        r_x_app, r_z_app = self.rotation_matricies(point_b)

        s_est = r_x_est @ r_z_est @ initial_state
        s_app = r_x_app @ r_z_app @ initial_state

        p_correct = (np.abs(s_app.T @ s_est) ** 2)[0,0]

        return p_correct

    def next_experiment(self): 
        
        curr_risk = float('inf')
        bayes_mean = self.calc_bayes_mean()
        self.prev_measurement = self.curr_measurement
            
        
        if self.n_qubits == 1:
            range_max = 2
        else:
            range_max = self.n_qubits ** 2
        
        # Apply Gaussian Noise
        tmp_dist = Distribution(n_points=self.n_points, n_qubits=self.n_qubits)
        tmp_dist.weights = self.weights
        tmp_dist.points = self.points
        tmp_dist.resample()

        # Select top N performing points
        unitaries = list(zip(tmp_dist.weights, tmp_dist.points))
        sorted(unitaries, key=lambda x: x[0])
        unitaries = np.array(unitaries[:self.points_to_sample])

        for weight, curr_unitary in unitaries:
            
            # Create new distributions   
            average_risk = 0
            for j in range(range_max):
                
                measurement_result = list(map(int, list(bin(j)[2:])))
                measurement_result = [0] * (self.n_qubits - len(measurement_result)) + measurement_result
                measurement_result = np.array(measurement_result)

                prior_probability = np.product([
                    self.bloch_distance(unitary, mean) 
                    if self.bloch_distance(unitary, mean) == result 
                    else  1 - self.bloch_distance(unitary, mean) 
                        for unitary, result, mean in zip(
                            np.array(curr_unitary).reshape(len(curr_unitary) // 2, 2),
                            measurement_result,
                            np.array(bayes_mean).reshape(len(bayes_mean) // 2, 2)
                        )
                    ]
                    )
                
                curr_distribution = Distribution(n_points = self.n_points) 
                curr_distribution.n_qubits = self.n_qubits
                curr_distribution.n_points = self.n_points
                curr_distribution.points = copy.deepcopy(self.points)
                curr_distribution.weights = copy.deepcopy(self.weights)
                curr_distribution.inversion_arr = curr_unitary
        
                # Speculate on the measurements
                curr_distribution.measure(measurement_result, curr_unitary, resample=False)
        
                # Correct up to normalisation
                average_risk += prior_probability * curr_distribution.calc_bayes_risk()
          
            
            if average_risk < curr_risk:
                curr_risk = abs(average_risk)
                self.curr_measurement = curr_unitary
            
        return self.curr_measurement