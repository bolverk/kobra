"""
Test suite for the brute force module
"""

import unittest
import numpy

class TestSuite(unittest.TestCase):

    """
    Test suite for this module
    """

    def test_pivot_to_generator(self):

        """
        Verifies the conversion of a pivot vector to a generator
        """

        from numpy.random import rand
        from brute_force import pivot2generator

        pivot = rand(3)
        generator = pivot2generator(pivot)
        for i in range(3):
            self.assertEqual(generator[i, i], 0)
        for i in range(3):
            for j in range(i):
                self.assertEqual(generator[i, j], -generator[j, i])
        temp = numpy.dot(generator, pivot)
        for i in range(3):
            self.assertEqual(temp[i], 0)

    def test_pivot_to_rotation(self):

        """
        Verifies the conversion of a pivot vector to a rotation matrix
        """

        from brute_force import pivot2rotation
        from numpy.random import rand

        pivot = rand(3)
        rot = pivot2rotation(pivot)
        temp = numpy.identity(3) - numpy.dot(numpy.transpose(rot), rot)
        res = 0
        for i in range(3):
            for j in range(3):
                res += temp[i, j]**2
        self.assertTrue(res < 1e-10)

    def test_circular_motion_t2ma(self):

        """
        Verifies the conversion of time to mean anomaly, for the secular case of circular motion
        """

        from brute_force import convert_time2mean_anomaly
        from numpy.random import rand

        for _ in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':rand(),
                   'eccentricity':0,
                   'periapse time':0,
                   'pivot':[0, 0, 0]}
            time = rand()
            mean_anomaly = convert_time2mean_anomaly(time, kop)
            timescale = 1/numpy.sqrt(kop['GM']/kop['semilatus rectum']**3)
            self.assertAlmostEqual(mean_anomaly, time/timescale)

    def test_circular_motion_ma2t(self):

        """
        Verifies the conversion of mean anomaly to time for the secular case of circular motion
        """

        from brute_force import convert_mean_anomaly2time
        from numpy.random import rand

        for _ in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':rand(),
                   'eccentricity':0,
                   'periapse time':0,
                   'pivot':[0, 0, 0]}
            mean_anomaly = rand()
            time = convert_mean_anomaly2time(mean_anomaly, kop)
            timescale = 1/numpy.sqrt(kop['GM']/kop['semilatus rectum']**3)
            self.assertAlmostEqual(mean_anomaly, time/timescale)

    def test_ma2t_reciprocity(self):

        """
        Verifies the conversion back and forth between mean and true anomaly
        """

        from brute_force import convert_mean_anomaly2time
        from brute_force import convert_time2mean_anomaly
        from numpy.random import rand

        for _ in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':rand(),
                   'eccentricity':rand(),
                   'periapse time':rand(),
                   'pivot':[0, 0, 0]}
            mean_anomaly = rand()
            time = convert_mean_anomaly2time(mean_anomaly, kop)
            reconstructed_mean_anomaly = convert_time2mean_anomaly(time, kop)
            self.assertAlmostEqual(mean_anomaly, reconstructed_mean_anomaly)

    def test_trajectory_consistency(self):

        """
        Verifies the velocity and position generated are consistent
        """

        from brute_force import generate_complete_trajectory
        from brute_force import mid_array
        from numpy.random import rand

        kop = {'GM':1,
               'semilatus rectum':0.5*(1+rand()),
               'eccentricity':rand(),
               'periapse time':rand(),
               'pivot':rand(3)}
        time_list = numpy.linspace(0, 10, 10000)
        ctr = generate_complete_trajectory(kop, time_list)
        derivs = {
            'vx':numpy.diff(ctr['position'].T[0])/numpy.diff(time_list),
            'vy':numpy.diff(ctr['position'].T[1])/numpy.diff(time_list),
            'vz':numpy.diff(ctr['position'].T[2])/numpy.diff(time_list)}
        mid = {'vx':mid_array(ctr['velocity'].T[0]),
               'vy':mid_array(ctr['velocity'].T[1]),
               'vz':mid_array(ctr['velocity'].T[2])}
        for velc in mid:
            for itm1, itm2 in zip(mid[velc], derivs[velc]):
                self.assertAlmostEqual(itm1, itm2, places=4)

    def test_chi_square_eval(self):

        """
        Verifies that the minimum value of chi squared
        is only obtained for the original Keplerian orbit parameters
        """

        from brute_force import generate_astrometry
        from brute_force import eval_chi_2
        from numpy.random import rand

        ref_kop = {'GM':1.0,
                   'semilatus rectum':rand(),
                   'eccentricity':0.2*rand(),
                   'periapse time':rand(),
                   'pivot':5*rand(3)}
        time_list = numpy.linspace(0, 10, 100)
        astrometry = generate_astrometry(ref_kop, time_list)
        ref_chi_2 = eval_chi_2(ref_kop, astrometry)
        for _ in range(10):
            kop = {'GM':1.0,
                   'semilatus rectum':rand(),
                   'eccentricity':0.2*rand(),
                   'periapse time':rand(),
                   'pivot':5*rand(3)}
            chi_2 = eval_chi_2(kop, astrometry)
            self.assertTrue(chi_2 > ref_chi_2)

    def test_estimate_init_params(self):

        """
        Verifies the estimation of the initial keplerian parameters
        """

        from brute_force import generate_astrometry
        from brute_force import estimate_initial_parameters
        from numpy.random import rand

        kop = {'GM':1.0,
               'semilatus rectum':1.0,
               'eccentricity':0.5,
               'periapse time':0.2,
               'pivot':rand(3)}
        time_list = numpy.linspace(0, 10, 10000)
        astrometry = generate_astrometry(kop, time_list)
        sol = estimate_initial_parameters(astrometry, grp=kop['GM'])

        self.assertAlmostEqual(sol['semilatus rectum'],
                               kop['semilatus rectum'],
                               places=4)
        self.assertAlmostEqual(sol['eccentricity'],
                               kop['eccentricity'],
                               places=4)
        self.assertAlmostEqual(sol['periapse time'],
                               kop['periapse time'],
                               places=4)
        for i in range(3):
            self.assertAlmostEqual(sol['pivot'][i],
                                   kop['pivot'][i],
                                   places=4)

    def test_brute_force_fit(self):

        """
        Verifies that the brute force fit reproduces the Keplerian parameters
        """

        from brute_force import generate_astrometry
        from brute_force import fit_parameters_bf

        kop = {'GM':1,
               'semilatus rectum':1.5,#rand(),
               'eccentricity':0.3,#0.9*rand(),
               'periapse time':0.1,#rand(),
               'pivot':numpy.array([0.1, -0.2, 0.3])}#rand(3)}
        time_list = numpy.linspace(0, 10, 100)
        astrometry = generate_astrometry(kop, time_list)
        sol = fit_parameters_bf(astrometry)
        self.assertAlmostEqual(sol['semilatus rectum'],
                               kop['semilatus rectum'],
                               places=1)
        self.assertAlmostEqual(sol['eccentricity'],
                               kop['eccentricity'],
                               places=2)
        self.assertAlmostEqual(sol['periapse time'],
                               kop['periapse time'],
                               places=2)

    def test_rotation_fit(self):

        """
        Verifies that the rotation based parameter fit reproduces the original Keplerian parameters
        """

        from brute_force import generate_astrometry
        from brute_force import fit_parameters_wr

#        kop = {'GM':1,
#               'semilatus rectum':rand(),
#               'eccentricity':0.9*rand(),
#               'periapse time':rand(),
#               'pivot':rand(3)}
        kop = {'GM':1,
               'semilatus rectum':1.5,#rand(),
               'eccentricity':0.3,#0.9*rand(),
               'periapse time':0.1,#rand(),
               'pivot':numpy.array([0.1, -0.2, 0.3])}#rand(3)}
        time_list = numpy.linspace(0, 10, 100)
        astrometry = generate_astrometry(kop, time_list)
        sol = fit_parameters_wr(astrometry)
        self.assertAlmostEqual(sol['semilatus rectum'],
                               kop['semilatus rectum'],
                               places=1)
        self.assertAlmostEqual(sol['eccentricity'],
                               kop['eccentricity'],
                               places=2)
        self.assertAlmostEqual(sol['periapse time'],
                               kop['periapse time'],
                               places=2)

    def test_best_cayley_rotation(self):

        """
        Verifies that the Cayley rotation function reproduces the pivot vector
        """

        from numpy.random import rand
        from brute_force import pivot2rotation
        from brute_force import calc_best_cayley_rotation

        x_list = rand(100, 3)
        pivot = rand(3)
        rotation = pivot2rotation(pivot)
        y_list = numpy.dot(rotation, x_list.T).T
        reproduced = calc_best_cayley_rotation(x_list, y_list)
        for itm1, itm2 in zip(pivot, reproduced):
            self.assertAlmostEqual(itm1, itm2)

    def test_calc_gl_fit(self):

        """
        Verifies the calibration of a linear
        transformation that aligns the theoretical and observational data
        """

        from numpy.random import rand
        from brute_force import generate_complete_trajectory
        from brute_force import pivot2rotation
        from brute_force import generate_astrometry
        from brute_force import calc_gl_position_block
        from brute_force import calc_gl_velocity_block

        kop = {'GM':1,
               'semilatus rectum':rand(),
               'eccentricity':rand(),
               'periapse time':rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0, 10, 1000)
        trj = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = rand(3)
        rotation = pivot2rotation(kop['pivot'])
        amd = generate_astrometry(kop, time_list)
        blocks = {'position':calc_gl_position_block(amd, trj),
                  'velocity':calc_gl_velocity_block(amd, trj)}
        for i in range(2):
            self.assertAlmostEqual(blocks['velocity'][i], rotation[2, i])
            for j in range(2):
                self.assertAlmostEqual(blocks['position'][i, j], rotation[i, j])

    def test_gen_pivot_reciprocity(self):

        """
        Verifies the conversion back and forth between anti symmetric generator and pivot vector
        """

        from numpy.random import rand
        from brute_force import pivot2generator
        from brute_force import generator2pivot

        pivot = rand(3)
        generator = pivot2generator(pivot)
        reproduced = generator2pivot(generator)
        for itm1, itm2 in zip(pivot, reproduced):
            self.assertAlmostEqual(itm1, itm2)

    def test_calc_pivot_from_gl_block(self):

        """
        Verifies the calculation of a pivot vector from a general linear transformation
        """

        from numpy.random import rand
        from brute_force import pivot2rotation
        from brute_force import calc_pivot_from_gl_block

        pivot = rand(3)
        rotation = pivot2rotation(pivot)
        proj = numpy.zeros((2, 3))
        proj[0, 0] = 1
        proj[1, 1] = 1
        block = numpy.dot(rotation, proj.T)
        reproduced = calc_pivot_from_gl_block(block)
        for itm1, itm2 in zip(pivot, reproduced):
            self.assertAlmostEqual(itm1, itm2)

    def test_rot_pivot_reciprocity(self):

        """
        Verifies the conversion back and forth between pivot vector and rotation matrix
        """

        from numpy.random import rand
        from brute_force import pivot2rotation
        from brute_force import rotation2pivot

        pivot = rand(3)
        rotation = pivot2rotation(pivot)
        reproduced = rotation2pivot(rotation)
        for itm1, itm2 in zip(pivot, reproduced):
            self.assertAlmostEqual(itm1, itm2)

    def test_fit_small_rotation(self):

        """
        Verifies the calculation of rotation matrix, in the limit of small rotation angles
        """

        from numpy.random import rand
        from brute_force import generate_complete_trajectory
        from brute_force import generate_astrometry
        from brute_force import fit_small_rotation

        kop = {'GM':1,
               'semilatus rectum':rand(),
               'eccentricity':rand(),
               'periapse time':rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0, 10, 1000)
        trj = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = 1e-4*rand(3)
        amd = generate_astrometry(kop, time_list)
        reproduced = fit_small_rotation(amd, trj)
        for itm1, itm2 in zip(kop['pivot'], reproduced):
            self.assertAlmostEqual(itm1, itm2)

    def test_fit_rotation(self):

        """
        Verifies the calculation of a rotation matrix, without assuming small rotation angles
        """

        from numpy.random import rand
        from brute_force import generate_complete_trajectory
        from brute_force import generate_astrometry
        from brute_force import fit_rotation_to_astrometry

        kop = {'GM':1,
               'semilatus rectum':rand(),
               'eccentricity':rand(),
               'periapse time':rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0, 10, 1000)
        trj = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = rand(3)
        amd = generate_astrometry(kop, time_list)
        reproduced = fit_rotation_to_astrometry(amd, trj, n_itr=3)
        for itm1, itm2 in zip(kop['pivot'], reproduced):
            self.assertAlmostEqual(itm1, itm2)

if __name__ == '__main__':

    unittest.main()
