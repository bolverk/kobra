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

    res = numpy.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i==(j+1)%3:
                res[i,j] = pivot[3-i-j]
            if j==(i+1)%3:
                res[i,j] = -pivot[3-i-j]
    return res
    
def generator2rotation(generator):

    g = generator
    g2 = numpy.dot(g,g)
    return numpy.identity(3)+2.0*(g+g2)/(1-0.5*numpy.trace(g2))
    
def pivot2rotation(pivot):

    return generator2rotation(pivot2generator(pivot))
    
def generate_complete_trajectory(kop, time_list):

    mean_anomalies = convert_time2mean_anomaly(time_list,kop)
    #eccentric_anomalies = numpy.array([eccentric_anomaly_from_mean(kop['eccentricity'],m,tolerance=abs(1e-7*m)) for m in mean_anomalies])
    eccentric_anomalies = numpy.array([solve_kepler_equation(kop['eccentricity'],m) for m in mean_anomalies])
    true_anomalies = numpy.array([true_anomaly_from_eccentric(kop['eccentricity'],E) for E in eccentric_anomalies])
    cos = numpy.cos
    sin = numpy.sin
    sqrt = numpy.sqrt
    GM = kop['GM']
    rl = kop['semilatus rectum']
    e = kop['eccentricity']
    a = rl/(1-e**2) # Semi major axis
    radius_list = numpy.array([orbit_radius(a,e,f) for f in true_anomalies])
    #radius_list = numpy.array([rl/(1+e*cos(f)) for f in true_anomalies])
    position_face_on = numpy.array([r*numpy.array([cos(q),sin(q),0]) for r,q in zip(radius_list,true_anomalies)])
    rotation = pivot2rotation(kop['pivot'])
    position_list = numpy.array([numpy.dot(rotation,r) for r in position_face_on])
    velocity_face_on = numpy.array([sqrt(GM/rl)*numpy.array([-sin(q),e+cos(q),0]) for q in true_anomalies])
    velocity_list = numpy.array([numpy.dot(rotation,v) for v in velocity_face_on])
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
        
    temp = minimize(func,[1,0,0,0,0,0],bounds=[(1e-6,100),(0,0.9999),(None,None),(None,None),(None,None),(None,None)])

    return unfold_data(temp.x)
    
def eval_rotation_chi_2(astrometry, ellipse_params):
        
    kop = {}
    for field in ellipse_params:
        kop[field] = ellipse_params[field]
    kop['pivot'] = numpy.zeros(3)
    trajectory = generate_complete_trajectory(kop,astrometry['t'])
        
    s2d_list = trajectory['position'].T[:2].T
    r2d_list = numpy.vstack((astrometry['x'],astrometry['y'])).T
    mat_list = [numpy.outer(s,s) for s in s2d_list]
    mat_1 = numpy.zeros((2,2))
    for m in mat_list:
        mat_1 += m
    mat_list = [numpy.outer(r,s) for r,s in zip(r2d_list,s2d_list)]
    mat_2 = numpy.zeros((2,2))
    for m in mat_list:
        mat_2 += m
    position_block = numpy.dot(mat_2,numpy.linalg.inv(mat_1))
        
    v2d_list = trajectory['velocity'].T[:2].T
    #mat_1 = numpy.sum([numpy.outer(v,v) for v in v2d_list])
    mat_list = [numpy.outer(v,v) for v in v2d_list]
    mat_1 = numpy.zeros((2,2))
    for m in mat_list:
        mat_1 += m
    vec_array = [w*v for w,v in zip(astrometry['vz'],v2d_list)]
    vec_1 = numpy.zeros(2)
    for v in vec_array:
        vec_1 += v
    #vec_1 = numpy.sum([w*v for w,v in zip(astrometry['vz'],v2d_list)],axis=1)
    velocity_block = numpy.dot(numpy.linalg.inv(mat_1),vec_1)
        
    col_vec_1 = numpy.array([position_block[0,0],position_block[1,0],velocity_block[0]])
    col_vec_2 = numpy.array([position_block[0,1],position_block[1,1],velocity_block[1]])
        
    return (numpy.dot(col_vec_1,col_vec_1)-1)**2+(numpy.dot(col_vec_2,col_vec_2)-1)**2+(numpy.dot(col_vec_1,col_vec_2))**2

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
    logger.debug(str(fsp['time'])+' '+str(kop['periapse time'])+' '+
                 str(mean_motion)+' '+str(q)+' '+
                 str(E)+' '+str(M))
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
    res = {field:numpy.average([kop[field] for kop in kop_list]) for field in kop_list[0]}
    return res

def fit_parameters_wr(astrometry,GM=1):

    from scipy.optimize import minimize

    def unfold_data(x):
    
        kop = {'GM':GM,
               'semilatus rectum':x[0],
               'eccentricity':x[1],
               'periapse time':x[2],
               'pivot':[0,0,0]}
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
               'pivot':numpy.zeros(3)}
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

    def testBruteForceParameterFit(self):
        
        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':numpy.random.rand(),
               'periapse time':numpy.random.rand(),
               'pivot':numpy.random.rand(3)}
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
    
if __name__ == '__main__':

    unittest.main()
