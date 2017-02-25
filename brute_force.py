"""
This modules calculates keplerian orbit parameters from observational data
"""

import unittest
import numpy
from orbital.utilities import eccentric_anomaly_from_true, true_anomaly_from_eccentric
from orbital.utilities import mean_anomaly_from_eccentric, eccentric_anomaly_from_mean
from orbital.utilities import orbit_radius
from sympy import Eijk

def eval_levi_civita_tensor():

    """
    Evaluates the Levi Civita tensor

    :return: Levi Civita tensor
    """

    return numpy.array([[[Eijk(i, j, k) for k in range(3)]
                         for j in range(3)]
                        for i in range(3)],
                       dtype=numpy.float)

# Levi Civita Symbol
LCS = eval_levi_civita_tensor()

def sqr_norm(vec):

    """
    Calculates the square of the norm of a vector.
    Basically I wanted to avoid writing the same vector twice.

    :param v: Vector
    :return: Scalar
    """

    return numpy.dot(vec, vec)

def calc_best_cayley_rotation(x_list, y_list):

    """
    Given two sets of ordered/labelled vectors,
    finds the rotation matrix that brings them close together

    :param x_list: First list of vectors
    :param y_list: Second list of vectors
    :return: Pivot vector
    """

    mat = (numpy.identity(3)*numpy.sum(
        sqr_norm(x+y) for x, y in zip(x_list, y_list)) -
           numpy.sum(numpy.kron(x+y, x+y) for
                     x, y in zip(x_list, y_list)).reshape((3, 3)))
    vec = numpy.sum((numpy.cross(x, y) for x, y in zip(x_list, y_list)))

    return 2*numpy.dot(numpy.linalg.inv(mat), vec)

def solve_kepler_equation(eccentricity, mean_anomaly):

    """
    Calculates the eccentric anomaly by solving Kepler's equation.

    :param e: Eccentricity
    :param M: Mean anomaly
    :return: Eccentric anomaly (a scalar)
    """

    from scipy.optimize import fsolve

    sin = numpy.sin

    def eval_kepler_equation(eccentric_anomaly):

        """
        Evaluates the difference between the two sides of the kepler equation

        :param eccentric_anomaly: eccentric anomaly
        :return: Distance from a solution of the Kepler equation
        """

        return (mean_anomaly - eccentric_anomaly +
                eccentricity*sin(eccentric_anomaly))

    return fsolve(eval_kepler_equation, mean_anomaly)

def convert_mean_anomaly2time(mean_anomaly, kop):

    """
    Converts mean anomaly to time.
    This is the reverse of
    :py:meth:`brute_force.convert_time2mean_anomaly`

    :param mean_anomaly: Mean anomaly
    :param kop: Keplerian orbit parameters
    :return: Time (scalar)
    """

    sqrt = numpy.sqrt
    ecc = kop['eccentricity']
    top = kop['periapse time']
    slr = kop['semilatus rectum']
    grp = kop['GM']
    return top + mean_anomaly/(1-ecc**2)**1.5/sqrt(grp/slr**3)

def convert_time2mean_anomaly(time, kop):

    """
    Converts time to mean anomaly.
    This is the reverse of
    :py:meth:`brute_force.convert_mean_anomaly2time`

    :param time: Time
    :param kop: Keplerian orbit parameters
    :return: Mean anomaly (scalar)
    """

    sqrt = numpy.sqrt
    ecc = kop['eccentricity']
    top = kop['periapse time']
    slr = kop['semilatus rectum']
    grp = kop['GM']
    return (time-top)*sqrt(grp/slr**3)*(1-ecc**2)**1.5

def pivot2generator(pivot):

    """
    Maps a three dimensional vector to an
    anti-symmetric matrix by tensor multiplication
    with the Levi - Civita Symbol.
    This is the reverse of
    :py:meth:`brute_force.generator2pivot`

    :param pivot: Pivot vector
    :return: Anti - symmetric matrix
    """

    return -numpy.einsum('ijk,k', LCS, pivot)

def calc_pivot_from_gl_block(block):

    """
    Calculates the rotation based on the information
    from the rectangular matrix block given from the astrometric fit.

    :param block: General linear 2x3 matrix
    :return: Pivot vector
    """

    proj = numpy.zeros((2, 3))
    proj[0, 0] = 1
    proj[1, 1] = 1
    aux = numpy.dot(block+proj.T, (block+proj.T).T)
    mat = numpy.identity(3)*numpy.trace(aux)-aux
    vec = generator2pivot(numpy.dot(block, proj))
    return 4*numpy.dot(numpy.linalg.inv(mat), vec)

def generator2pivot(generator):

    """
    Maps an anti - symmetric matrix to a pivot vector.
    This is the reverse of :py:meth:`brute_force.pivot2generator`

    :param generator: An anti symmetric matrix
    :return: Pivot vector
    """

    return -0.5*numpy.einsum('ijk,jk', LCS, generator)

def generator2rotation(generator):

    """
    Calculates a rotation matrix from an anti symmetric generator using the Cayley transformation

    :param generator: Anti - symmetric generator matrix
    :return: Rotation matrix
    """

    gen = generator
    gen2 = numpy.dot(gen, gen)
    return numpy.identity(3)+2.0*(gen+gen2)/(1-0.5*numpy.trace(gen2))

def pivot2rotation(pivot):

    """
    Calculates the rotation matrix from a pivot vector.
    This is the reverse of :py:meth:`brute_force.rotation2pivot`

    :param pivot: Pivot vector
    :return: Rotation matrix
    """

    return generator2rotation(pivot2generator(pivot))

def rotation2pivot(rotation):

    """
    Calculates the pivot vector from a rotation.
    This is the reverse of :py:meth:`brute_force.pivot2rotation`

    :param rotation: Rotation matrix
    :return: Pivot vector
    """

    id3 = numpy.identity(3)
    aux = numpy.dot(rotation+id3,
                    (rotation+id3).T)
    return 4*numpy.dot(
        numpy.linalg.inv(id3*numpy.trace(aux)-aux),
        generator2pivot(rotation))

def generate_complete_trajectory(kop, time_list):

    """
    Creates a list of positions and velocities of an object along a Keplerian orbit

    :param kop: Keplerian orbit parameters
    :param time_list: List of times when the positions and velocities will be evaluated
    :return: positions and velocities
    """

    mean_anomalies = convert_time2mean_anomaly(time_list, kop)
    eccentric_anomalies = numpy.fromiter(
        (solve_kepler_equation(kop['eccentricity'], m)
         for m in mean_anomalies), dtype=numpy.float)
    true_anomalies = numpy.fromiter(
        (true_anomaly_from_eccentric(kop['eccentricity'], E)
         for E in eccentric_anomalies), dtype=numpy.float)
    grp = kop['GM']
    slr = kop['semilatus rectum']
    ecc = kop['eccentricity']
    sma = slr/(1-ecc**2) # Semi major axis
    radius_list = numpy.fromiter((orbit_radius(sma, ecc, f)
                                  for f in true_anomalies),
                                 dtype=numpy.float)
    position_face_on = numpy.vstack((
        radius_list*numpy.cos(true_anomalies),
        radius_list*numpy.sin(true_anomalies),
        numpy.zeros_like(radius_list))).T
    rotation = pivot2rotation(kop['pivot'])
    position_list = numpy.dot(rotation, position_face_on.T).T

    velocity_face_on = numpy.vstack((
        -numpy.sqrt(grp/slr)*numpy.sin(true_anomalies),
        numpy.sqrt(grp/slr)*(ecc+numpy.cos(true_anomalies)),
        numpy.zeros_like(true_anomalies))).T
    velocity_list = numpy.dot(rotation, velocity_face_on.T).T
    return {'position':position_list, 'velocity':velocity_list}

def generate_astrometry(kop, time_list):

    """
    Simulates observational data.

    :param kop: Keplerian orbit parameters
    :param time_list: List of observation times
    :return: astrometry
    """

    trajectory = generate_complete_trajectory(kop, time_list)

    return {'t':time_list,
            'x':trajectory['position'].T[0],
            'y':trajectory['position'].T[1],
            'vz':trajectory['velocity'].T[2]}

def eval_chi_2(kop, astrometry):

    """
    Evaluates how well a certain Keplerian orbit fits observational data

    :param kop: Keplerian orbit parameters
    :param astrometry: Astrometric data (proper motion and radial velocity)
    :return: Chi squared value
    """

    reproduction = generate_astrometry(kop, astrometry['t'])

    huge_number = 1e9
    if kop['semilatus rectum'] < 0:
        return huge_number
    if kop['eccentricity'] > 1 or kop['eccentricity'] < 0:
        return huge_number

    res = 0
    for field in ['x', 'y', 'vz']:
        res += numpy.sum((astrometry[field]-reproduction[field])**2)
    res /= len(astrometry['x'])

    return res

def mid_array(arr):

    """
    Linearly interpolates values between consecutive entries

    :param arr: Original array
    :return: Array whose length is shorter by 1 from the length of ar
    """

    return 0.5*(arr[1:]+arr[:-1])

def fit_small_rotation(astrometry, trajectory):

    """
    Finds a rotation that best aligns a complete
    theoretical trajectory with observed data,
    assuming the necessary rotation is small
    (rotation angle much smaller than 1)

    :param astrometry: Astrometric data (proper motion and radial velocity)
    :param trajectory: Complete trajectory
    :return: Pivot vector
    """

    wmat = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    zmat = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    s_list = trajectory['position']
    r_list = numpy.vstack((astrometry['x'],
                           astrometry['y'],
                           numpy.zeros_like(astrometry['x']))).T
    es_list = numpy.einsum('ijk,lk', LCS, s_list).T
    u_list = trajectory['velocity']
    v_list = numpy.vstack((numpy.zeros_like(astrometry['vz']),
                           numpy.zeros_like(astrometry['vz']),
                           astrometry['vz'])).T
    eu_list = numpy.einsum('ijk,lk', LCS, u_list).T
    mat1 = -numpy.einsum('ijk,kl,ilm', es_list, wmat, es_list)/len(es_list)
    mat2 = -numpy.einsum('ijk,kl,ilm', eu_list, zmat, eu_list)/len(eu_list)
    vec1 = numpy.einsum('ijk,jl,nl,nk', LCS, wmat, r_list-s_list, s_list)/len(s_list)
    vec2 = numpy.einsum('ijk,jl,nl,nk', LCS, zmat, v_list-u_list, u_list)/len(u_list)

    return -0.5*numpy.dot(numpy.linalg.inv(mat1+mat2),
                          vec1+vec2)

def fit_rotation_to_astrometry(astrometry, trajectory, n_itr=3):

    """
    Finds the rotation necessary to align a theoretical
    trajectory with observations.

    :param astrometry: Astrometric data (proper motion and radial velocity)
    :param trajectory: Complete theoretical trajectory
    :param n_itr: Number of iteration
    :return: Pivot vector
    """

    block = numpy.vstack((
        calc_gl_position_block(astrometry, trajectory),
        calc_gl_velocity_block(astrometry, trajectory)))
    pivot = calc_pivot_from_gl_block(block)
    for _ in range(n_itr):
        rot = pivot2rotation(pivot)
        new_traj = {'position':numpy.dot(rot, trajectory['position'].T).T,
                    'velocity':numpy.dot(rot, trajectory['velocity'].T).T}
        change = fit_small_rotation(astrometry, new_traj)
        drot = pivot2rotation(change)
        pivot = rotation2pivot(numpy.dot(drot, rot))
    return pivot

def fit_parameters_bf(astrometry, grp=1):

    """
    Fits Keplerian parameters to astrometric data
    using a "brute force" approach, i.e. fits all 6
    parameters simultaneously.

    :param astrometry: Astrometric data (proper motion and radial velocity)
    :param GM: Gravitational parameter
    :return: Keplerian orbit parameters
    """

    from scipy.optimize import minimize

    def unfold_data(arglist):

        """
        Arranges data in convenient form

        :param arglist: List of arguments
        :return: Arguments in a dictionary
        """

        kop = {'GM':grp,
               'semilatus rectum':arglist[0],
               'eccentricity':arglist[1],
               'periapse time':arglist[2],
               'pivot':arglist[3:]}
        return kop

    def func(arglist):

        """
        Evaluates chi squared

        :param arglist: Argument list
        :return: Value of chi squared
        """

        kop = unfold_data(arglist)
        return eval_chi_2(kop, astrometry)

    ipg = estimate_initial_parameters(astrometry)
    temp = minimize(func, [ipg['semilatus rectum'],
                           ipg['eccentricity'],
                           ipg['periapse time'],
                           ipg['pivot'][0],
                           ipg['pivot'][1],
                           ipg['pivot'][2]],
                    bounds=[(1e-6, 100),
                            (0, 0.9999),
                            (None, None),
                            (None, None),
                            (None, None),
                            (None, None)])

    return unfold_data(temp.x)

def eval_rotation_chi_2(astrometry, ellipse_params):

    """
    Evaluates how well a certain elliptical orbit fits
    observational data. The difference from
    :py:meth:`brute_force.eval_chi_2` is that this
    function figures out the rotation on its own.

    :param astrometry: Observation data
    :param ellipse_params: Same as keplerian orbit parameters, but without the rotation pivot
    :return: Value of chi squared
    """

    kop = {}
    for field in ellipse_params:
        kop[field] = ellipse_params[field]
    kop['pivot'] = numpy.zeros(3)
    trajectory = generate_complete_trajectory(kop, astrometry['t'])
    pivot = fit_rotation_to_astrometry(astrometry, trajectory)
    kop['pivot'] = pivot
    #return eval_chi_2(kop,astrometry)
    rotation = pivot2rotation(pivot)
    for field in ['position', 'velocity']:
        trajectory[field] = numpy.dot(rotation, trajectory[field].T).T
    res = 0
    res += numpy.sum((astrometry['x']-trajectory['position'].T[0])**2)
    res += numpy.sum((astrometry['y']-trajectory['position'].T[1])**2)
    res += numpy.sum((astrometry['vz']-trajectory['velocity'].T[2])**2)
    return res

def energy_from_phase_space_point(fsp, grp=1):

    """
    Calculates the energy from a position and velocity

    :param fsp: Phase space point. A dictionary containing the position and velocity
    :param GM: Gravitational parameter
    :return: Energy
    """

    rad = numpy.linalg.norm(fsp['position'])
    return -grp/rad+0.5*sqr_norm(fsp['velocity'])

def angular_momentum_from_psp(psp):

    """
    Calculates the angular momentum from a position and velocity

    :param fsp: Phase space point. A dictionary containing the position and velocity
    :param GM: Gravitational parameter
    :return: Angular momentum vector
    """

    return numpy.cross(psp['position'], psp['velocity'])

def mean_anomaly_from_true(ecc, tra):

    """
    Calculates the mean anomaly from the true anomaly

    :param ecc: Eccentricity
    :param tra: True anomaly
    :return: Mean anomaly
    """

    eca = eccentric_anomaly_from_true(ecc, tra)
    return mean_anomaly_from_eccentric(ecc, eca)

def orbital_parameters_from_psp(fsp, grp=1):

    """
    Reproduces the Keplerian orbital parameters from a position and velocity

    :param fsp: Phase space point. A dictionary containing the position and velocity
    :param GM: Gravitational parameter
    :return: Keplerian orbit parameters
    """

    def slr_eccentricity_formula():
        enr = energy_from_phase_space_point(fsp, grp)
        anm = angular_momentum_from_psp(fsp)
        ecc = numpy.sqrt(1+2*sqr_norm(anm)*enr/grp**2)
        slr = sqr_norm(anm)/grp
        return slr, ecc
    slr, ecc = slr_eccentricity_formula()
    def true_anomaly_formula():
        rad = numpy.linalg.norm(fsp['position'])
        cosq = (slr/rad-1)
        sinq = numpy.dot(fsp['position'],
                         fsp['velocity'])/numpy.sqrt(grp/slr)/rad
        return numpy.arctan2(sinq, cosq)
    q = true_anomaly_formula()
    M = mean_anomaly_from_true(ecc, q)
    kop = {'semilatus rectum':slr,
           'eccentricity':ecc,
           'periapse time':0,
           'GM':grp}
    dt = convert_mean_anomaly2time(M, kop)
    def normalise_periapse_time():
        res = fsp['time'] - dt
        mean_motion = 1.0/numpy.sqrt((1-ecc**2)**3*grp/slr**3)
        period = 2*numpy.pi*mean_motion
        while res < 0:
            res += period
        while res > period:
            res -= period
        return res
    kop['periapse time'] = normalise_periapse_time()
    a = slr/(1-ecc**2)
    r_2d = orbit_radius(a, ecc, q)*numpy.array(
        [numpy.cos(q), numpy.sin(q), 0])
    v_2d = numpy.sqrt(grp/slr)*numpy.array(
        [-numpy.sin(q), ecc+numpy.cos(q), 0])
    kop['pivot'] = calc_best_cayley_rotation(numpy.vstack((r_2d, v_2d)),
                                             numpy.vstack((fsp['position'],
                                                           fsp['velocity'])))
    return kop

def estimate_initial_parameters(astrometry, GM=1):

    """
    Estimates the keplerian orbit parameters. Used as initial guess for the minimisation.

    :param astrometry: Astrometric data (proper motion and radial velocity)
    :param GM: Gravitational parameter
    :return: Keplerian orbit parameters
    """

    vx = numpy.diff(astrometry['x'])/numpy.diff(astrometry['t'])
    vy = numpy.diff(astrometry['y'])/numpy.diff(astrometry['t'])
    vz_mid = mid_array(astrometry['vz'])
    t_mid = mid_array(astrometry['t'])
    ax = numpy.diff(vx)/numpy.diff(t_mid)
    ay = numpy.diff(vy)/numpy.diff(t_mid)
    az = numpy.diff(vz_mid)/numpy.diff(t_mid)
    x_mid_mid = mid_array(mid_array(astrometry['x']))
    y_mid_mid = mid_array(mid_array(astrometry['y']))
    r_mid_mid = numpy.sqrt(GM/numpy.sqrt(ax**2+ay**2+az**2))
    z_mid_mid = -az*r_mid_mid**3/GM
    t_mid_mid = mid_array(t_mid)
    vx_mid_mid = mid_array(vx)
    vy_mid_mid = mid_array(vy)
    vz_mid_mid = mid_array(vz_mid)
    r_list = numpy.vstack((x_mid_mid, y_mid_mid, z_mid_mid)).T
    v_list = numpy.vstack((vx_mid_mid, vy_mid_mid, vz_mid_mid)).T
    kop_list = [orbital_parameters_from_psp(
        {'position':r, 'velocity':v, 'time':t})
                for r, v, t in zip(r_list, v_list, t_mid_mid)]
    res = {field:numpy.average([kop[field] for kop in kop_list])
           for field in kop_list[0] if not field == 'pivot'}
    res['pivot'] = numpy.sum((kop['pivot'] for kop in kop_list))/len(kop_list)
    return res

def calc_gl_position_block(astrometry, trajectory):

    """
    Finds the best linear transformation that
    aligns the theoretical trajectory to the proper motion data

    :param astrometry: Observational data (proper motion and radial velocity)
    :param trajectory: Complete theoretical trajectory
    :return: A 2x2 matrix
    """

    s2d_list = trajectory['position'].T[:2].T
    r2d_list = numpy.vstack((astrometry['x'], astrometry['y'])).T
    mat_1 = numpy.einsum('ni,nj', s2d_list, s2d_list)
    mat_2 = numpy.einsum('ni,nj', r2d_list, s2d_list)

    return numpy.dot(mat_2, numpy.linalg.inv(mat_1))

def calc_gl_velocity_block(astrometry, trajectory):

    """
    Finds the best linear transformation that
    aligns the theoretical trajectory to the proper radial velocity data

    :param astrometry: Observational data (proper motion and radial velocity)
    :param trajectory: Complete theoretical trajectory
    :return: A 2 vector
    """

    v2d_list = trajectory['velocity'].T[:2].T
    mat_1 = numpy.einsum('ni,nj', v2d_list, v2d_list)
    vec_1 = numpy.einsum('n,ni', astrometry['vz'], v2d_list)
    return numpy.dot(numpy.linalg.inv(mat_1), vec_1)

def fit_parameters_wr(astrometry, GM=1):

    """
    Fits Keplerian orbital parameters to observational data.
    Uses calculation of the rotation matrix for optimisation.

    :param astrometry: Observational data (proper motion and radial velocity)
    :param GM: Gravitational parameter
    :return: Keplerian orbtial parameters
    """

    from scipy.optimize import minimize

    def unfold_data(x):

        kop = {'GM':GM,
               'semilatus rectum':x[0],
               'eccentricity':x[1],
               'periapse time':x[2],
               'pivot':numpy.zeros(3)}
        return kop

    def func(x):

        kop = unfold_data(x)
        return eval_rotation_chi_2(astrometry, kop)

    initial_guess = estimate_initial_parameters(astrometry, GM)
    rl = initial_guess['semilatus rectum']
    e = initial_guess['eccentricity']
    t0 = initial_guess['periapse time']

    temp = minimize(func, [rl, e, t0],
                    bounds=
                    [(1e-6, 100),
                     (0, 0.99),
                     (None, None)])

    return unfold_data(temp.x)

"""
Tests
"""

class TestSuite(unittest.TestCase):

    """
    Test suite for this module
    """

    def testPivot2Generator(self):

        """
        Verifies the conversion of a pivot vector to a generator
        """

        pivot = numpy.random.rand(3)
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

        pivot = numpy.random.rand(3)
        R = pivot2rotation(pivot)
        temp = numpy.identity(3) - numpy.dot(numpy.transpose(R), R)
        res = 0
        for i in range(3):
            for j in range(3):
                res += temp[i, j]**2
        self.assertTrue(1e-10 > res)

    def testTime2MeanAnomalyConversionCircularMotion(self):

        """
        Verifies the conversion of time to mean anomaly, for the secular case of circular motion
        """

        for i in range(1000):
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

        for i in range(1000):
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

        for i in range(1000):
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

        ref_kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0.2*numpy.random.rand(),
                   'periapse time':numpy.random.rand(),
                   'pivot':5*numpy.random.rand(3)}
        time_list = numpy.linspace(0, 10, 100)
        astrometry = generate_astrometry(ref_kop, time_list)
        ref_chi_2 = eval_chi_2(ref_kop, astrometry)
        for i in range(10):
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

        kop = {'GM':1.0,
               'semilatus rectum':1.0,
               'eccentricity':0.5,
               'periapse time':0.2,
               'pivot':numpy.random.rand(3)}
        time_list = numpy.linspace(0, 10, 10000)
        ct = generate_complete_trajectory(kop, time_list)
        astrometry = generate_astrometry(kop, time_list)
        sol = estimate_initial_parameters(astrometry, GM=kop['GM'])

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

        x_list = numpy.random.rand(100, 3)
        pivot = numpy.random.rand(3)
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

        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0, 10, 1000)
        ct = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = numpy.random.rand(3)
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

        pivot = numpy.random.rand(3)
        generator = pivot2generator(pivot)
        reproduced = generator2pivot(generator)
        for a, b in zip(pivot, reproduced):
            self.assertAlmostEqual(a, b)

    def testCalcPivotFromGLRectangleBlock(self):

        """
        Verifies the calculation of a pivot vector from a general linear transformation
        """

        pivot = numpy.random.rand(3)
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

        pivot = numpy.random.rand(3)
        rotation = pivot2rotation(pivot)
        reproduced = rotation2pivot(rotation)
        for a, b in zip(pivot, reproduced):
            self.assertAlmostEqual(a, b)

    def testFitSmallRotation(self):

        """
        Verifies the calculation of rotation matrix, in the limit of small rotation angles
        """

        from time import time

        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0, 10, 1000)
        ct = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = 1e-4*numpy.random.rand(3)
        ad = generate_astrometry(kop, time_list)
        reproduced = fit_small_rotation(ad, ct)
        for a, b in zip(kop['pivot'], reproduced):
            self.assertAlmostEqual(a, b)

    def testFitRotation(self):

        """
        Verifies the calculation of a rotation matrix, without assuming small rotation angles
        """

        from time import time

        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0, 10, 1000)
        ct = generate_complete_trajectory(kop, time_list)
        kop['pivot'] = numpy.random.rand(3)
        ad = generate_astrometry(kop, time_list)
        reproduced = fit_rotation_to_astrometry(ad, ct, n_itr=3)
        for a, b in zip(kop['pivot'], reproduced):
            self.assertAlmostEqual(a, b)

if __name__ == '__main__':

    unittest.main()
