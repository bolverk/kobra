import unittest
import numpy

class TestSuite(unittest.TestCase):

    """
    Test suite for this module
    """

    def testPivot2Generator(self):

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

    def testPivot2Rotation(self):

        """
        Verifies the conversion of a pivot vector to a rotation matrix
        """

        from brute_force import pivot2rotation

        pivot = numpy.random.rand(3)
        R = pivot2rotation(pivot)
        temp = numpy.identity(3) - numpy.dot(numpy.transpose(R), R)
        res = 0
        for i in range(3):
            for j in range(3):
                res += temp[i, j]**2
        self.assertTrue(res < 1e-10)

    def testTime2MeanAnomalyConversionCircularMotion(self):

        """
        Verifies the conversion of time to mean anomaly, for the secular case of circular motion
        """

        from brute_force import convert_time2mean_anomaly

        for _ in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0,
                   'periapse time':0,
                   'pivot':[0, 0, 0]}
            time = numpy.random.rand()
            mean_anomaly = convert_time2mean_anomaly(time, kop)
            timescale = 1/numpy.sqrt(kop['GM']/kop['semilatus rectum']**3)
            self.assertAlmostEqual(mean_anomaly, time/timescale)

    def testMeanAnomaly2TimeConversionCircularMotion(self):

        """
        Verifies the conversion of mean anomaly to time for the secular case of circular motion
        """

        from brute_force import convert_mean_anomaly2time

        for _ in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0,
                   'periapse time':0,
                   'pivot':[0, 0, 0]}
            mean_anomaly = numpy.random.rand()
            time = convert_mean_anomaly2time(mean_anomaly, kop)
            timescale = 1/numpy.sqrt(kop['GM']/kop['semilatus rectum']**3)
            self.assertAlmostEqual(mean_anomaly, time/timescale)

    def testMeanAnomalyTimeReciprocity(self):

        """
        Verifies the conversion back and forth between mean and true anomaly
        """

        from brute_force import convert_mean_anomaly2time
        from brute_force import convert_time2mean_anomaly

        for _ in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':numpy.random.rand(),
                   'periapse time':numpy.random.rand(),
                   'pivot':[0, 0, 0]}
            mean_anomaly = numpy.random.rand()
            time = convert_mean_anomaly2time(mean_anomaly, kop)
            reconstructed_mean_anomaly = convert_time2mean_anomaly(time, kop)
            self.assertAlmostEqual(mean_anomaly, reconstructed_mean_anomaly)

    def testTrajectoryPositionVelocityConsistency(self):

        """
        Verifies the velocity and position generated are consistent
        """

        from brute_force import generate_complete_trajectory
        from brute_force import mid_array

        kop = {'GM':1,
               'semilatus rectum':0.5*(1+numpy.random.rand()),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.random.rand(3)}
        time_list = numpy.linspace(0, 10, 10000)
        ct = generate_complete_trajectory(kop, time_list)
        dxdt_list = numpy.diff(ct['position'].T[0])/numpy.diff(time_list)
        dydt_list = numpy.diff(ct['position'].T[1])/numpy.diff(time_list)
        dzdt_list = numpy.diff(ct['position'].T[2])/numpy.diff(time_list)
        vx_mid = mid_array(ct['velocity'].T[0])
        vy_mid = mid_array(ct['velocity'].T[1])
        vz_mid = mid_array(ct['velocity'].T[2])
        for dxdt, vx in zip(dxdt_list, vx_mid):
            self.assertAlmostEqual(vx, dxdt, places=4)
        for dydt, vy in zip(dydt_list, vy_mid):
            self.assertAlmostEqual(vy, dydt, places=4)
        for dzdt, vz in zip(dzdt_list, vz_mid):
            self.assertAlmostEqual(vz, dzdt, places=4)

    def testChiSquareEvaluations(self):

        """
        Verifies that the minimum value of chi squared
        is only obtained for the original Keplerian orbit parameters
        """

        from brute_force import generate_astrometry
        from brute_force import eval_chi_2

        ref_kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0.2*numpy.random.rand(),
                   'periapse time':numpy.random.rand(),
                   'pivot':5*numpy.random.rand(3)}
        time_list = numpy.linspace(0, 10, 100)
        astrometry = generate_astrometry(ref_kop, time_list)
        ref_chi_2 = eval_chi_2(ref_kop, astrometry)
        for _ in range(10):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0.2*numpy.random.rand(),
                   'periapse time':numpy.random.rand(),
                   'pivot':5*numpy.random.rand(3)}
            chi_2 = eval_chi_2(kop, astrometry)
            self.assertTrue(chi_2 > ref_chi_2)

    def testEstimateInitialParameters(self):

        """
        Verifies the estimation of the initial keplerian parameters
        """

        from brute_force import generate_astrometry
        from brute_force import estimate_initial_parameters

        kop = {'GM':1.0,
               'semilatus rectum':1.0,
               'eccentricity':0.5,
               'periapse time':0.2,
               'pivot':numpy.random.rand(3)}
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

    def testBruteForceParameterFit(self):

        """
        Verifies that the brute force fit reproduces the Keplerian parameters
        """

        from brute_force import generate_astrometry
        from brute_force import fit_parameters_bf

        kop = {'GM':1,
               'semilatus rectum':1.5,#numpy.random.rand(),
               'eccentricity':0.3,#0.9*numpy.random.rand(),
               'periapse time':0.1,#numpy.random.rand(),
               'pivot':numpy.array([0.1, -0.2, 0.3])}#numpy.random.rand(3)}
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

    def testRotationParameterFit(self):

        """
        Verifies that the rotation based parameter fit reproduces the original Keplerian parameters
        """

        from brute_force import generate_astrometry
        from brute_force import fit_parameters_wr

#        kop = {'GM':1,
#               'semilatus rectum':numpy.random.rand(),
#               'eccentricity':0.9*numpy.random.rand(),
#               'periapse time':numpy.random.rand(),
#               'pivot':numpy.random.rand(3)}
        kop = {'GM':1,
               'semilatus rectum':1.5,#numpy.random.rand(),
               'eccentricity':0.3,#0.9*numpy.random.rand(),
               'periapse time':0.1,#numpy.random.rand(),
               'pivot':numpy.array([0.1, -0.2, 0.3])}#numpy.random.rand(3)}
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

    def testBestCayleyRotation(self):

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
        for p1, p2 in zip(pivot, reproduced):
            self.assertAlmostEqual(p1, p2)

    def testCalcGLFit(self):

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
        ct = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = rand(3)
        rotation = pivot2rotation(kop['pivot'])
        ad = generate_astrometry(kop, time_list)
        position_block = calc_gl_position_block(ad, ct)
        velocity_block = calc_gl_velocity_block(ad, ct)
        for i in range(2):
            self.assertAlmostEqual(velocity_block[i], rotation[2, i])
            for j in range(2):
                self.assertAlmostEqual(position_block[i, j], rotation[i, j])

    def testGeneratorPivotReciprocity(self):

        """
        Verifies the conversion back and forth between anti symmetric generator and pivot vector
        """

        from numpy.random import rand
        from brute_force import pivot2generator
        from brute_force import generator2pivot

        pivot = rand(3)
        generator = pivot2generator(pivot)
        reproduced = generator2pivot(generator)
        for a, b in zip(pivot, reproduced):
            self.assertAlmostEqual(a, b)

    def testCalcPivotFromGLRectangleBlock(self):

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
        for a, b in zip(pivot, reproduced):
            self.assertAlmostEqual(a, b)

    def testRotation2PivotReciprocity(self):

        """
        Verifies the conversion back and forth between pivot vector and rotation matrix
        """

        from numpy.random import rand
        from brute_force import pivot2rotation
        from brute_force import rotation2pivot

        pivot = rand(3)
        rotation = pivot2rotation(pivot)
        reproduced = rotation2pivot(rotation)
        for a, b in zip(pivot, reproduced):
            self.assertAlmostEqual(a, b)

    def testFitSmallRotation(self):

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
        ct = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = 1e-4*rand(3)
        ad = generate_astrometry(kop, time_list)
        reproduced = fit_small_rotation(ad, ct)
        for a, b in zip(kop['pivot'], reproduced):
            self.assertAlmostEqual(a, b)

    def testFitRotation(self):

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
        ct = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = rand(3)
        ad = generate_astrometry(kop, time_list)
        reproduced = fit_rotation_to_astrometry(ad, ct, n_itr=3)
        for a, b in zip(kop['pivot'], reproduced):
            self.assertAlmostEqual(a, b)

if __name__ == '__main__':

    unittest.main()
