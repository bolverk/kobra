from kobra import estimate_rtbp_parameters
import numpy
import unittest

class TestSuite(unittest.TestCase):

    """
    Test suite for this module
    """

    def test_guess_parameters(self):

        from kobra import guess_proper_motion
        from kobra import generate_observational_data
        from brute_force import pivot2rotation
        from kobra import guess_lz_over_d2
        from kobra import guess_angular_momentum_ratios
        from kobra import guess_hodograph
        from kobra import hodograph2physical_params

        rtbpp = {'alpha 0':1e-4*numpy.random.rand(),
                 'beta 0':1e-4*numpy.random.rand(),
                 'eccentricity':0.2,
                 'periapse time':10,
                 'semilatus rectum':1,
                 'GM':4.5e-8,
                 'pivot':numpy.random.rand(3),
                 'distance':1e4,
                 'dot alpha 0':2.5e-8,
                 'dot beta 0':1e-8,
                 'w 0':1e-4}
        t_list = numpy.linspace(0,50,2e3)
        obs = generate_observational_data(rtbpp, t_list)
        pmp = guess_proper_motion(obs)
        field_list = ['alpha 0',
                      'dot alpha 0',
                      'beta 0',
                      'dot beta 0']
        aux = [rtbpp[field] for field in field_list]
        for itm1, itm2 in zip(pmp,aux):
            self.assertAlmostEqual(itm1,itm2,places=3)
        l_mag = numpy.sqrt(rtbpp['GM']*rtbpp['semilatus rectum'])
        rot = pivot2rotation(rtbpp['pivot'])
        lz_over_d2 = guess_lz_over_d2(obs)
        self.assertAlmostEqual(lz_over_d2,rot[2,2]*l_mag/rtbpp['distance']**2)
        l_ratios = guess_angular_momentum_ratios(obs)
        self.assertAlmostEqual(l_ratios[0], rtbpp['w 0'], places=3)
        self.assertAlmostEqual(l_ratios[1]/rtbpp['distance'],
                               rot[0,2]/rot[2,2],
                               places=3)
        self.assertAlmostEqual(l_ratios[2]/rtbpp['distance'],
                               rot[1,2]/rot[2,2],
                               places=3)
        hodograph_raw = guess_hodograph(obs)
        hodograph_data = hodograph2physical_params(
            hodograph_raw,
            lz_over_d2,
            l_ratios)
        self.assertAlmostEqual(hodograph_data['distance']/
                               rtbpp['distance'],
                               1,
                               places=3)
        self.assertAlmostEqual(hodograph_data['angular momentum'][2],
                               l_mag*rot[2,2],
                               places=3)
        self.assertAlmostEqual(hodograph_data['angular momentum'][1],
                               l_mag*rot[1,2],
                               places=3)
        self.assertAlmostEqual(hodograph_data['angular momentum'][0],
                               l_mag*rot[0,2],
                               places=3)

    def testEstimateRTBPParameters(self):

        from kobra import generate_observational_data

        rtbpp = {'alpha 0':1e-4*numpy.random.rand(),
                 'beta 0':1e-4*numpy.random.rand(),
                 'eccentricity':0.2,
                 'periapse time':10,
                 'semilatus rectum':1,
                 'GM':4.5e-8,
                 'pivot':numpy.random.rand(3),
                 'distance':1e4,
                 'dot alpha 0':2.5e-8,
                 'dot beta 0':1e-8,
                 'w 0':1e-4}
        t_list = numpy.linspace(0,50,1e3)
        od = generate_observational_data(rtbpp, t_list)
        reproduced = estimate_rtbp_parameters(od)
        for vname in reproduced:
            if not vname=='distance':
                self.assertAlmostEqual(reproduced[vname],
                                       rtbpp[vname],
                                       places=3)
            if vname=='distance':
                val1 = reproduced[vname]
                val2 = rtbpp[vname]
                self.assertTrue(abs(val1-val2)/abs(val1+val2)<0.01)