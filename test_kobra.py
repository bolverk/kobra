from kobra import estimate_rtbp_parameters
import numpy
import unittest

def diff_rat(num_1, num_2):

    return abs(num_1-num_2)/(abs(num_1)+abs(num_2))

def calc_vector_angle(vec1, vec2, pvt):

    from math import copysign

    cosq = numpy.dot(vec1,vec2)
    v1cv2 = numpy.cross(vec1, vec2)
    sinq = copysign(
        numpy.linalg.norm(v1cv2),
        numpy.dot(pvt,v1cv2))
    
    return numpy.arctan2(sinq,cosq)

class TestSuite(unittest.TestCase):

    """
    Test suite for this module
    """

    def test_guess_parameters(self):

        from kobra import guess_proper_motion
        from kobra import generate_observational_data
        from brute_force import pivot2rotation
        from kobra import guess_angular_momentum_ratios
        from kobra import guess_hodograph
        from kobra import hodograph2physical_params
        from brute_force import mean_anomaly_from_true
        from brute_force import convert_mean_anomaly2time

        rtbpp = {'alpha 0':1e-4,
                 'beta 0':-1e-4,
                 'eccentricity':0.2,
                 'periapse time':1000.0,
                 'semilatus rectum':0.2,
                 'GM':4.5e-8,
                 'pivot':numpy.array([1,-2,3]),
                 'distance':1e4,
                 'dot alpha 0':2.5e-8,
                 'dot beta 0':1e-8,
                 'w 0':1e-4}
        t_list = numpy.linspace(0,5000,1e4)
        obs = generate_observational_data(rtbpp, t_list)
        pmp = guess_proper_motion(obs)
        field_list = ['alpha 0',
                      'beta 0',
                      'dot alpha 0',
                      'dot beta 0']
        for field in ['alpha 0',
                      'beta 0',
                      'dot alpha 0',
                      'dot beta 0']:
            self.assertTrue(
                diff_rat(
                    rtbpp[field],
                    pmp[field])<1e-7)
        l_mag = numpy.sqrt(rtbpp['GM']*rtbpp['semilatus rectum'])
        rot = pivot2rotation(rtbpp['pivot'])
        lz_over_d2 = pmp['lz/d**2']
        self.assertTrue(
            diff_rat(
                lz_over_d2,
                rot[2,2]*l_mag/
                rtbpp['distance']**2)<1e-7)
        l_ratios = guess_angular_momentum_ratios(obs)
        self.assertTrue(
            diff_rat(l_ratios[0],
                     rtbpp['w 0'])<1e-7)
        self.assertTrue(
            diff_rat(l_ratios[1]/rtbpp['distance'],
                     rot[0,2]/rot[2,2])<1e-7)
        self.assertTrue(
            diff_rat(l_ratios[2]/rtbpp['distance'],
                     rot[1,2]/rot[2,2])<1e-7)
        hodograph_raw = guess_hodograph(obs)
        hodograph_data = hodograph2physical_params(
            hodograph_raw,
            lz_over_d2,
            l_ratios)
        self.assertTrue(
            diff_rat(hodograph_data['distance'],
                     rtbpp['distance'])<1e-7)
        self.assertTrue(
            diff_rat(hodograph_data['angular momentum'][2],
                     l_mag*rot[2,2])<1e-7)
        self.assertTrue(
            diff_rat(hodograph_data['angular momentum'][1],
                     l_mag*rot[1,2])<1e-7)
        self.assertTrue(
            diff_rat(hodograph_data['angular momentum'][0],
                     l_mag*rot[0,2])<1e-7)
        self.assertTrue(
            diff_rat(
                numpy.dot(
                    hodograph_data['edotmu'],
                    hodograph_data['edotmu']),
                (rtbpp['GM']*rtbpp['eccentricity'])**2)<1e-7)
        self.assertTrue(
            diff_rat(
                hodograph_data['mu'],
                rtbpp['GM'])<1e-7)
        self.assertTrue(
            diff_rat(
                hodograph_data['eccentricity'],
                rtbpp['eccentricity'])<1e-7)
        for itm1, itm2 in zip(hodograph_data['pivot'],rtbpp['pivot']):
            self.assertTrue(
                diff_rat(itm1,itm2)<1e-7)
        reproduced_slr = numpy.linalg.norm(
            hodograph_data['angular momentum'])**2/hodograph_data['mu']
        self.assertTrue(
            diff_rat(reproduced_slr,rtbpp['semilatus rectum'])<1e-7)
        x_list = hodograph_data['distance']*(
            obs['alpha']-pmp['alpha 0']-pmp['dot alpha 0']*obs['t'])
        y_list = hodograph_data['distance']*(
            obs['beta']-pmp['beta 0']-pmp['dot beta 0']*obs['t'])
        z_list = -(
            (x_list*hodograph_data['angular momentum'][0]+
             y_list*hodograph_data['angular momentum'][1])/
            hodograph_data['angular momentum'][2])
        q_list = numpy.array(
            [calc_vector_angle(
                [x,y,z],
                hodograph_data['eccentricity vector'],
                -hodograph_data['angular momentum'])
             for x,y,z in zip(x_list,y_list,z_list)])
        m_list = numpy.array(
            [mean_anomaly_from_true(hodograph_data['eccentricity'],q)
             for q in q_list])
        t0_array = obs['t']-numpy.array(
            [convert_mean_anomaly2time(m,{
                'eccentricity':hodograph_data['eccentricity'],
                'semilatus rectum':reproduced_slr,
                'GM':hodograph_data['mu'],
                'periapse time':0
            })
             for m in m_list])
        mean_motion = 1.0/numpy.sqrt(
            (1-hodograph_data['eccentricity']**2)**3*
            hodograph_data['mu']/reproduced_slr**3)
        period = 2*numpy.pi*mean_motion
        for i in range(len(t0_array)):
            while t0_array[i]<0:
                t0_array[i] += period
            while t0_array[i]>period:
                t0_array[i] -= period
        reproduced_t0 = numpy.average(t0_array)
        self.assertTrue(
            diff_rat(reproduced_t0,
                     rtbpp['periapse time'])<1e-7)
                  

    def testEstimateRTBPParameters(self):

        from kobra import generate_observational_data

        rtbpp = {'alpha 0':1e-4,
                 'beta 0':-1e-4,
                 'eccentricity':0.2,
                 'periapse time':1000.0,
                 'semilatus rectum':0.2,
                 'GM':4.5e-8,
                 'pivot':numpy.array([1,-2,3]),
                 'distance':1e4,
                 'dot alpha 0':2.5e-8,
                 'dot beta 0':1e-8,
                 'w 0':1e-4}
        t_list = numpy.linspace(0,50,1e3)
        od = generate_observational_data(rtbpp, t_list)
        reproduced = estimate_rtbp_parameters(od)
        for vname in reproduced:
            self.assertTrue(
                diff_rat(rtbpp[vname],
                         reproduced[vname])<1e-7)
            
