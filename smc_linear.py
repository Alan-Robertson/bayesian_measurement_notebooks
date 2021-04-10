import random
import math
import numpy as np
import copy


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
                 resampler_h=np.sqrt(0.1)):
        
        
        self.vals_per_qubit = 2
        if distribution_generator is not None:
            self.points, self.weights = distribution_generator(n_points, n_qubits)
        else:
            self.points, self.weights = self.random_distribution(n_points=n_points, n_qubits=n_qubits)
        
        self.resampler_a = resampler_a
        self.resampler_h = resampler_h
        
        self.n_qubits = n_qubits
        self.n_points = len(self.weights)
        self.rejection_threshold = rejection_threshold
        self.covariance_matrix = None
        
        
        
        # Inversion array, flips then measures
        self.inversion_arr = np.zeros(self.n_qubits, dtype=int)
        
    def measure(self, measurement_data, measurement_target, inversion_arr=None, resample=True):
        
        if resample:
            self.conditional_resample()
        
        if inversion_arr is None:
            inversion_arr = self.inversion_arr
        
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
        
        
    # Replace this bias with the covariance
    def update_estimate(self, point, weight, target_arr, measurement_outcome):
        '''
            Calculate the new weight from the previous one
        '''        
        total_weight = weight
                
        for i, (state, target) in enumerate(zip(measurement_outcome, target_arr)):
            
            # Points are paired
            value = point[i * 2 + target]
            
            if state == target:
                total_weight *= value / self.n_qubits / pow(self.n_qubits, 0.5)
            else:
                total_weight *= (1 - value) / self.n_qubits / pow(self.n_qubits, 0.5)
                
        return total_weight

    def calc_bayes_mean(self):
        bayes_mean = 0

        for point, weight in zip(self.points, self.weights):
            bayes_mean += point * weight
        return bayes_mean
            
        
    def random_distribution(self, n_points, start=0, stop=0.25, n_qubits=1):
        # Bias the fuck out of this initial distribution
        points = []
        
        for _ in range(n_points):
            points.append(np.random.rand(self.vals_per_qubit * n_qubits) * (stop - start) + start)
        
            
        weights = np.array([1 / len(points)] * len(points))
        points = np.array(points)
        
        return points, weights
        
    
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
                if val < 0:
                    self.points[i][j] = np.sqrt(abs(val))
                if val > 1:
                    self.points[i][j] = np.sqrt(abs(1 - val))
        
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

    
    def next_experiment(self): 
        
        curr_risk = self.calc_bayes_risk()
        curr_measurement = self.inversion_arr
        bayes_mean = self.calc_bayes_mean()
        
        for i in range(self.n_qubits):
            curr_inversion_arr = [0] * self.n_qubits
            curr_inversion_arr[i] = 1
            
            # Create new distributions   
            average_risk = 0
            for j in range(self.n_qubits):
                
                measurement_result = [0] * self.n_qubits
                measurement_result[i] = 1
                
                prior_probabilities = np.product([bayes_mean[inv] if inv == res else 1 - bayes_mean[inv] for inv, res, mean in zip(
                    curr_inversion_arr,
                    measurement_result,
                    bayes_mean)])
                
                n_flipped = sum(measurement_result)
                n_uneffected = len(measurement_result) - n_flipped
                
                curr_distribution = Distribution(n_points = self.n_points) 
                curr_distribution.n_qubits = self.n_qubits
                curr_distribution.n_points = self.n_points
                curr_distribution.points = copy.deepcopy(self.points)
                curr_distribution.weights = copy.deepcopy(self.weights)
                curr_distribution.inversion_arr = curr_inversion_arr
        
                # Speculate on the measurements
                curr_distribution.measure(measurement_result, curr_inversion_arr, resample=False)
        
                # Correct up to normalisation
                average_risk += prior_probabilities * curr_distribution.calc_bayes_risk()
          
            
            
            if average_risk < curr_risk:
                curr_risk = abs(average_risk)
                curr_measurement = curr_inversion_arr
        
        return curr_measurement