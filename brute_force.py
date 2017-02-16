import unittest
import numpy
from orbital.utilities import eccentric_anomaly_from_true, true_anomaly_from_eccentric
from orbital.utilities import mean_anomaly_from_eccentric, eccentric_anomaly_from_mean
from orbital.utilities import orbit_radius
import logging
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler('brute_force.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
from sympy import Eijk

# Levi Civita Symbol
lcs = numpy.array([[[Eijk(i,j,k) for k in range(3)]
                    for j in range(3)]
                   for i in range(3)], dtype=numpy.float)

def sqr_norm(v):

    return numpy.dot(v,v)

def calc_best_cayley_rotation(x_list, y_list):

    mat = numpy.identity(3)*numpy.sum(sqr_norm(x+y) for x,y in zip(x_list,y_list)) - numpy.sum(numpy.kron(x+y,x+y) for x,y in zip(x_list,y_list)).reshape((3,3))
    vec = numpy.sum((numpy.cross(x,y) for x,y in zip(x_list,y_list)))

    return 2*numpy.dot(numpy.linalg.inv(mat),vec)

def solve_kepler_equation(e,M):

    from scipy.optimize import fsolve

    sin = numpy.sin

    def eval_kepler_equation(E):
    
        return M - E + e*sin(E)
        
    return fsolve(eval_kepler_equation,M)
    
def convert_mean_anomaly2time(mean_anomaly, kop):

    sqrt = numpy.sqrt
    e = kop['eccentricity']
    t0 = kop['periapse time']
    rl = kop['semilatus rectum']
    GM = kop['GM']
    return t0 + mean_anomaly/(1-e**2)**1.5/sqrt(GM/rl**3)
    
def convert_time2mean_anomaly(time, kop):

    sqrt = numpy.sqrt
    e = kop['eccentricity']
    t0 = kop['periapse time']
    rl = kop['semilatus rectum']
    GM = kop['GM']
    return (time-t0)*sqrt(GM/rl**3)*(1-e**2)**1.5
    
def pivot2generator(pivot):

    return -numpy.einsum('ijk,k',lcs,pivot)

def calc_pivot_from_gl_rectangle_block(block):

    proj = numpy.zeros((2,3))
    proj[0,0] = 1
    proj[1,1] = 1
    aux = numpy.dot(block+proj.T,(block+proj.T).T)
    mat = numpy.identity(3)*numpy.trace(aux)-aux
    vec = generator2pivot(numpy.dot(block,proj))
    return 4*numpy.dot(numpy.linalg.inv(mat),vec)

def generator2pivot(generator):

    return -0.5*numpy.einsum('ijk,jk',lcs,generator)
    
def generator2rotation(generator):

    g = generator
    g2 = numpy.dot(g,g)
    return numpy.identity(3)+2.0*(g+g2)/(1-0.5*numpy.trace(g2))
    
def pivot2rotation(pivot):

    return generator2rotation(pivot2generator(pivot))

def rotation2pivot(rotation):

    i3 = numpy.identity(3)
    aux = numpy.dot(rotation+i3,
                    (rotation+i3).T)
    return 4*numpy.dot(
        numpy.linalg.inv(i3*numpy.trace(aux)-aux),
        generator2pivot(rotation))
    
def generate_complete_trajectory(kop, time_list):

    mean_anomalies = convert_time2mean_anomaly(time_list,kop)
    #eccentric_anomalies = numpy.array([eccentric_anomaly_from_mean(kop['eccentricity'],m,tolerance=abs(1e-7*m)) for m in mean_anomalies])
    eccentric_anomalies = numpy.fromiter((solve_kepler_equation(kop['eccentricity'],m) for m in mean_anomalies),dtype=numpy.float)
    true_anomalies = numpy.fromiter((true_anomaly_from_eccentric(kop['eccentricity'],E) for E in eccentric_anomalies),dtype=numpy.float)
    cos = numpy.cos
    sin = numpy.sin
    sqrt = numpy.sqrt
    GM = kop['GM']
    rl = kop['semilatus rectum']
    e = kop['eccentricity']
    a = rl/(1-e**2) # Semi major axis
    radius_list = numpy.fromiter((orbit_radius(a,e,f)
                                  for f in true_anomalies),
                                 dtype=numpy.float)
    #radius_list = numpy.array([rl/(1+e*cos(f)) for f in true_anomalies])
    x_face_on = radius_list*numpy.cos(true_anomalies)
    y_face_on = radius_list*numpy.sin(true_anomalies)
    z_face_on = numpy.zeros_like(radius_list)
    position_face_on = numpy.vstack((x_face_on,
                                     y_face_on,
                                     z_face_on)).T
    rotation = pivot2rotation(kop['pivot'])
    position_list = numpy.dot(rotation, position_face_on.T).T

    vx_face_on = -numpy.sqrt(GM/rl)*numpy.sin(true_anomalies)
    vy_face_on = numpy.sqrt(GM/rl)*(e+numpy.cos(true_anomalies))
    vz_face_on = numpy.zeros_like(true_anomalies)
    velocity_face_on = numpy.vstack((vx_face_on,
                                     vy_face_on,
                                     vz_face_on)).T
    velocity_list = numpy.dot(rotation,velocity_face_on.T).T
    return {'position':position_list,'velocity':velocity_list}
    
def generate_astrometry(kop,time_list):

    trajectory = generate_complete_trajectory(kop,time_list)
    
    return {'t':time_list,
            'x':trajectory['position'].T[0],
            'y':trajectory['position'].T[1],
            'vz':trajectory['velocity'].T[2]}
    
def eval_chi_2(kop, astrometry):

    reproduction = generate_astrometry(kop, astrometry['t'])
    
    huge_number = 1e9
    if kop['semilatus rectum']<0:
        return huge_number
    if kop['eccentricity']>1 or kop['eccentricity']<0:
        return huge_number
    
    
    res = 0
    for field in ['x','y','vz']:
        res += numpy.sum((astrometry[field]-reproduction[field])**2)
    res /= len(astrometry['x'])
    
    return res

def mid_array(ar):

    return 0.5*(ar[1:]+ar[:-1])

def fit_small_rotation(astrometry, trajectory):

    W = numpy.array([[1,0,0],[0,1,0],[0,0,0]])
    Z = numpy.array([[0,0,0],[0,0,0],[0,0,1]])
    s_list = trajectory['position']
    r_list = numpy.vstack((astrometry['x'],
                           astrometry['y'],
                           numpy.zeros_like(astrometry['x']))).T
    es_list = numpy.einsum('ijk,lk',lcs,s_list).T
    u_list = trajectory['velocity']
    v_list = numpy.vstack((numpy.zeros_like(astrometry['vz']),
                           numpy.zeros_like(astrometry['vz']),
                           astrometry['vz'])).T
    eu_list = numpy.einsum('ijk,lk',lcs,u_list).T
    m1 = -numpy.einsum('ijk,kl,ilm',es_list,W,es_list)/len(es_list)
    m2 = -numpy.einsum('ijk,kl,ilm',eu_list,Z,eu_list)/len(eu_list)
    v1 = numpy.einsum('ijk,jl,nl,nk',lcs,W,r_list-s_list,s_list)/len(s_list)
    v2 = numpy.einsum('ijk,jl,nl,nk',lcs,Z,v_list-u_list,u_list)/len(u_list)
    
    return -0.5*numpy.dot(numpy.linalg.inv(m1+m2),
                       v1+v2)

def fit_rotation_to_astrometry(astrometry, trajectory, n_itr=3):

    block = numpy.vstack((
        calc_gl_position_block(astrometry,trajectory),
        calc_gl_velocity_block(astrometry,trajectory)))
    pivot = calc_pivot_from_gl_rectangle_block(block)
    for i in range(n_itr):
        R = pivot2rotation(pivot)
        new_traj = {'position':numpy.dot(R,trajectory['position'].T).T,
                    'velocity':numpy.dot(R,trajectory['velocity'].T).T}
        change = fit_small_rotation(astrometry,new_traj)
        dR = pivot2rotation(change)
        pivot = rotation2pivot(numpy.dot(dR,R))
    return pivot

def fit_parameters_bf(astrometry,GM=1):

    from scipy.optimize import minimize
    
    def unfold_data(x):
    
        kop = {'GM':GM,
               'semilatus rectum':x[0],
               'eccentricity':x[1],
               'periapse time':x[2],
               'pivot':x[3:]}
        return kop

    def func(x):
    
        kop = unfold_data(x)
        return eval_chi_2(kop,astrometry)

    ig = estimate_initial_parameters(astrometry)
    temp = minimize(func,[ig['semilatus rectum'],
                          ig['eccentricity'],
                          ig['periapse time'],
                          ig['pivot'][0],
                          ig['pivot'][1],
                          ig['pivot'][2]],bounds=[(1e-6,100),(0,0.9999),(None,None),(None,None),(None,None),(None,None)])

    return unfold_data(temp.x)
    
def eval_rotation_chi_2(astrometry, ellipse_params):
        
    kop = {}
    for field in ellipse_params:
        kop[field] = ellipse_params[field]
    kop['pivot'] = numpy.zeros(3)
    trajectory = generate_complete_trajectory(kop,astrometry['t'])
    pivot = fit_rotation_to_astrometry(astrometry,
                                       trajectory)
    kop['pivot'] = pivot
    return eval_chi_2(kop,astrometry)    

def energy_from_phase_space_point(fsp,GM=1):

    r = numpy.linalg.norm(fsp['position'])
    return -GM/r+0.5*sqr_norm(fsp['velocity'])

def angular_momentum_from_phase_space_point(fsp):

    return numpy.cross(fsp['position'],fsp['velocity'])

def orbital_parameters_from_phase_space_point(fsp,GM=1):

    r = numpy.linalg.norm(fsp['position'])
    l = angular_momentum_from_phase_space_point(fsp)
    u = energy_from_phase_space_point(fsp,GM)
    rl = sqr_norm(l)/GM
    e = numpy.sqrt(1+2*sqr_norm(l)*u/GM**2)
    cosq = (rl/r-1)
    sinq = numpy.dot(fsp['position'],fsp['velocity'])/numpy.sqrt(GM/rl)/r
    q = numpy.arctan2(sinq,cosq)
    #q = numpy.arccos((rl/r-1)/e)
    E = eccentric_anomaly_from_true(e,q)
    M = mean_anomaly_from_eccentric(e,E)
    kop = {'semilatus rectum':rl,
           'eccentricity':e,
           'periapse time':0,
           'GM':GM}
    dt = convert_mean_anomaly2time(M,kop)
    kop['periapse time'] = fsp['time']-dt
    mean_motion = 1.0/numpy.sqrt((1-e**2)**3*GM/rl**3)
    period = 2*numpy.pi*mean_motion
    while kop['periapse time']<0:
        kop['periapse time'] += period
    while kop['periapse time'] > period:
        kop['periapse time'] -= period
    a = rl/(1-e**2)
    r_2d = orbit_radius(a,e,q)*numpy.array([numpy.cos(q),numpy.sin(q),0])
    v_2d = numpy.sqrt(GM/rl)*numpy.array([-numpy.sin(q),e+numpy.cos(q),0])
    kop['pivot'] =  calc_best_cayley_rotation(numpy.vstack((r_2d,v_2d)),
                                              numpy.vstack((fsp['position'],
                                                            fsp['velocity'])))
    return kop

def estimate_initial_parameters(astrometry, GM=1):

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
    r_list = numpy.vstack((x_mid_mid,y_mid_mid,z_mid_mid)).T
    v_list = numpy.vstack((vx_mid_mid,vy_mid_mid,vz_mid_mid)).T
    kop_list = [orbital_parameters_from_phase_space_point({'position':r,'velocity':v,'time':t}) for r,v,t in zip(r_list,v_list,t_mid_mid)]
    res = {field:numpy.average([kop[field] for kop in kop_list]) for field in kop_list[0] if not field=='pivot'}
    res['pivot'] = numpy.sum((kop['pivot'] for kop in kop_list))/len(kop_list)
    return res

def calc_gl_position_block(astrometry, trajectory):

    s2d_list = trajectory['position'].T[:2].T
    r2d_list = numpy.vstack((astrometry['x'],astrometry['y'])).T
    mat_1 = numpy.einsum('ni,nj',s2d_list,s2d_list)
    mat_2 = numpy.einsum('ni,nj',r2d_list,s2d_list)

    return numpy.dot(mat_2,numpy.linalg.inv(mat_1))

def calc_gl_velocity_block(astrometry, trajectory):

    v2d_list = trajectory['velocity'].T[:2].T
    mat_1 = numpy.einsum('ni,nj',v2d_list,v2d_list)
    vec_1 = numpy.einsum('n,ni',astrometry['vz'],v2d_list)
    return numpy.dot(numpy.linalg.inv(mat_1),vec_1)

def fit_parameters_wr(astrometry,GM=1):

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
        return eval_rotation_chi_2(astrometry,kop)

    initial_guess = estimate_initial_parameters(astrometry,GM)
    rl = initial_guess['semilatus rectum']
    e = initial_guess['eccentricity']
    t0 = initial_guess['periapse time']

    temp = minimize(func,[rl,e,t0],bounds=[(1e-6,100),(0,0.99),(None,None)])

    return unfold_data(temp.x)
            
"""
Tests
"""

class TestSuite(unittest.TestCase):

    def testPivot2Generator(self):
    
        pivot = numpy.random.rand(3)
        generator = pivot2generator(pivot)
        for i in range(3):
            self.assertEqual(generator[i,i],0)
        for i in range(3):
            for j in range(i):
                self.assertEqual(generator[i,j],-generator[j,i])
        temp = numpy.dot(generator,pivot)
        for i in range(3):
            self.assertEqual(temp[i],0)
            
    def testPivot2Rotation(self):
    
        pivot = numpy.random.rand(3)
        R = pivot2rotation(pivot)
        temp = numpy.identity(3) - numpy.dot(numpy.transpose(R),R)
        sum = 0
        for i in range(3):
            for j in range(3):
                sum += temp[i,j]**2
        self.assertTrue(1e-10>sum)
            
    def testTime2MeanAnomalyConversionCircularMotion(self):
    
        for i in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0,
                   'periapse time':0,
                   'pivot':[0,0,0]}
            time = numpy.random.rand()
            mean_anomaly = convert_time2mean_anomaly(time,kop)
            timescale = 1/numpy.sqrt(kop['GM']/kop['semilatus rectum']**3)
            self.assertAlmostEqual(mean_anomaly,time/timescale)
            
    def testMeanAnomaly2TimeConversionCircularMotion(self):
    
        for i in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0,
                   'periapse time':0,
                   'pivot':[0,0,0]}
            mean_anomaly = numpy.random.rand()
            time = convert_mean_anomaly2time(mean_anomaly,kop)
            timescale = 1/numpy.sqrt(kop['GM']/kop['semilatus rectum']**3)
            self.assertAlmostEqual(mean_anomaly,time/timescale)
            
    def testMeanAnomalyTimeReciprocity(self):
        for i in range(1000):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':numpy.random.rand(),
                   'periapse time':numpy.random.rand(),
                   'pivot':[0,0,0]}
            mean_anomaly = numpy.random.rand()
            time = convert_mean_anomaly2time(mean_anomaly,kop)
            reconstructed_mean_anomaly = convert_time2mean_anomaly(time,kop)
            self.assertAlmostEqual(mean_anomaly,reconstructed_mean_anomaly)

    def testTrajectoryPositionVelocityConsistency(self):

        kop = {'GM':1,
               'semilatus rectum':0.5*(1+numpy.random.rand()),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.random.rand(3)}
        time_list = numpy.linspace(0,10,10000)
        ct = generate_complete_trajectory(kop,time_list)
        dxdt_list = numpy.diff(ct['position'].T[0])/numpy.diff(time_list)
        dydt_list = numpy.diff(ct['position'].T[1])/numpy.diff(time_list)
        dzdt_list = numpy.diff(ct['position'].T[2])/numpy.diff(time_list)
        vx_mid = mid_array(ct['velocity'].T[0])
        vy_mid = mid_array(ct['velocity'].T[1])
        vz_mid = mid_array(ct['velocity'].T[2])
        for dxdt, vx in zip(dxdt_list,vx_mid):
            self.assertAlmostEqual(vx,dxdt,places=4)
        for dydt, vy in zip(dydt_list,vy_mid):
            self.assertAlmostEqual(vy,dydt,places=4)
        for dzdt, vz in zip(dzdt_list,vz_mid):
            self.assertAlmostEqual(vz,dzdt,places=4)
            
    def testChiSquareEvaluations(self):
    
        ref_kop = {'GM':1.0,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':0.2*numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':5*numpy.random.rand(3)}
        time_list = numpy.linspace(0,10,100)
        astrometry = generate_astrometry(ref_kop,time_list)
        ref_chi_2 = eval_chi_2(ref_kop,astrometry)
        for i in range(10):
            kop = {'GM':1.0,
                   'semilatus rectum':numpy.random.rand(),
                   'eccentricity':0.2*numpy.random.rand(),
                   'periapse time':numpy.random.rand(),
                   'pivot':5*numpy.random.rand(3)}
            chi_2 = eval_chi_2(kop,astrometry)
            self.assertTrue(chi_2>ref_chi_2)

    def testEstimateInitialParameters(self):

        kop = {'GM':1.0,
               'semilatus rectum':1.0,
               'eccentricity':0.5,
               'periapse time':0.2,
               'pivot':numpy.random.rand(3)}
        time_list = numpy.linspace(0,10,10000)
        ct = generate_complete_trajectory(kop,time_list)
        astrometry = generate_astrometry(kop,time_list)
        sol = estimate_initial_parameters(astrometry,GM=kop['GM'])

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
        
        kop = {'GM':1,
               'semilatus rectum':1.5,#numpy.random.rand(),
               'eccentricity':0.3,#0.9*numpy.random.rand(),
               'periapse time':0.1,#numpy.random.rand(),
               'pivot':numpy.array([0.1,-0.2,0.3])}#numpy.random.rand(3)}
        time_list = numpy.linspace(0,10,100)
        astrometry = generate_astrometry(kop,time_list)
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
        
#        kop = {'GM':1,
#               'semilatus rectum':numpy.random.rand(),
#               'eccentricity':0.9*numpy.random.rand(),
#               'periapse time':numpy.random.rand(),
#               'pivot':numpy.random.rand(3)}
        kop = {'GM':1,
               'semilatus rectum':1.5,#numpy.random.rand(),
               'eccentricity':0.3,#0.9*numpy.random.rand(),
               'periapse time':0.1,#numpy.random.rand(),
               'pivot':numpy.array([0.1,-0.2,0.3])}#numpy.random.rand(3)}
        time_list = numpy.linspace(0,10,100)
        astrometry = generate_astrometry(kop,time_list)
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

        x_list = numpy.random.rand(100,3)
        pivot = numpy.random.rand(3)
        rotation = pivot2rotation(pivot)
        y_list = numpy.dot(rotation,x_list.T).T
        reproduced = calc_best_cayley_rotation(x_list,y_list)
        for p1,p2 in zip(pivot,reproduced):
            self.assertAlmostEqual(p1,p2)

    def testCalcGLFit(self):

        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0,10,1000)
        ct = generate_complete_trajectory(kop,time_list)
        kop['pivot'] = numpy.random.rand(3)
        rotation = pivot2rotation(kop['pivot'])
        ad = generate_astrometry(kop,time_list)
        position_block = calc_gl_position_block(ad,ct)
        velocity_block = calc_gl_velocity_block(ad,ct)
        for i in range(2):
            self.assertAlmostEqual(velocity_block[i],rotation[2,i])
            for j in range(2):
                self.assertAlmostEqual(position_block[i,j],rotation[i,j])

    def testGeneratorPivotReciprocity(self):

        pivot = numpy.random.rand(3)
        generator = pivot2generator(pivot)
        reproduced = generator2pivot(generator)
        for a,b in zip(pivot,reproduced):
            self.assertAlmostEqual(a,b)

    def testCalcPivotFromGLRectangleBlock(self):

        pivot = numpy.random.rand(3)
        rotation = pivot2rotation(pivot)
        proj = numpy.zeros((2,3))
        proj[0,0] = 1
        proj[1,1] = 1
        block = numpy.dot(rotation,proj.T)
        reproduced = calc_pivot_from_gl_rectangle_block(block)
        for a,b in zip(pivot, reproduced):
            self.assertAlmostEqual(a,b)

    def testRotation2PivotReciprocity(self):

        pivot = numpy.random.rand(3)
        rotation = pivot2rotation(pivot)
        reproduced = rotation2pivot(rotation)
        for a,b in zip(pivot,reproduced):
            self.assertAlmostEqual(a,b)

    def testFitSmallRotation(self):

        from time import time

        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0,10,1000)
        ct = generate_complete_trajectory(kop,time_list)
        kop['pivot'] = 1e-4*numpy.random.rand(3)
        ad = generate_astrometry(kop,time_list)
        reproduced = fit_small_rotation(ad,ct)
        for a,b in zip(kop['pivot'],reproduced):
            self.assertAlmostEqual(a,b)

    def testFitRotation(self):

        from time import time

        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.zeros(3)}
        time_list = numpy.linspace(0,10,1000)
        ct = generate_complete_trajectory(kop,time_list)
        kop['pivot'] = numpy.random.rand(3)
        ad = generate_astrometry(kop,time_list)
        reproduced = fit_rotation_to_astrometry(ad,ct,n_itr=3)
        for a,b in zip(kop['pivot'],reproduced):
            self.assertAlmostEqual(a,b)
    
if __name__ == '__main__':

    unittest.main()
