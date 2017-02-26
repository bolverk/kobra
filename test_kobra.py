from kobra import estimate_rtbp_parameters
import numpy
import unittest

class TestSuite(unittest.TestCase):

    """
    Test suite for this module
    """

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
