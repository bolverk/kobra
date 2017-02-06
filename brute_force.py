import unittest
import numpy
from orbital.utilities import eccentric_anomaly_from_true, true_anomaly_from_eccentric
from orbital.utilities import mean_anomaly_from_eccentric, eccentric_anomaly_from_mean
from orbital.utilities import orbit_radius

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
    return t0 + sqrt(1-e**2)**1.5/sqrt(GM/rl**3)*mean_anomaly
    
def convert_time2mean_anomaly(time, kop):

    sqrt = numpy.sqrt
    e = kop['eccentricity']
    t0 = kop['periapse time']
    rl = kop['semilatus rectum']
    GM = kop['GM']
    return (time-t0)*sqrt(GM/rl**3)/sqrt(1-e**2)**1.5
    
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
    a = rl/(1-e) # Semi major axis
    radius_list = numpy.array([orbit_radius(a,e,f) for f in true_anomalies])
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
        
    rl_guess = numpy.sqrt(numpy.max(astrometry['x'])**2+numpy.max(astrometry['y'])**2)
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
        
    temp = minimize(func,[1,0,0],bounds=[(1e-6,100),(0,0.99),(None,None)])

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

    def testFaceOnCircle(self):
        
        kop = {'GM':1,
               'semilatus rectum':numpy.random.rand(),
               'eccentricity':0,
               'periapse time':1,
               'pivot':[7,-3,1]}
        time_list = numpy.linspace(0,10,100)
        astrometry = generate_astrometry(kop,time_list)
        sol = fit_parameters_wr(astrometry)
        self.assertAlmostEqual(sol['semilatus rectum'],kop['semilatus rectum'],places=4)
        self.assertAlmostEqual(sol['eccentricity'],kop['eccentricity'])
        self.assertAlmostEqual(sol['periapse time'],kop['periapse time'])
        
if __name__ == '__main__':

    unittest.main()