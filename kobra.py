import numpy

from brute_force import mid_array

def generate_observational_data(rtbpp, t_list):

    """
    Generates observational data
    
    :param rtbpp: Restricted two body problem parameters
    :param t_list: Time list
    :return: Observational data (astrometry and radial velocity)
    """

    from brute_force import generate_complete_trajectory

    ct = generate_complete_trajectory(rtbpp,t_list)
    
    #from mpl_toolkits.mplot3d import Axes3D
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(ct['velocity'].T[0],
    #           ct['velocity'].T[1],
    #           ct['velocity'].T[2])
    #ax.axis('equal')
    #plt.show()

    #import pylab
    #pylab.plot(ct['velocity'].T[0],
    #           ct['velocity'].T[1],'.')
    #pylab.axis('equal')
    #pylab.show()
    
    return {'t':t_list,
            'alpha':ct['position'].T[0]/rtbpp['distance']+
            rtbpp['alpha 0']+rtbpp['dot alpha 0']*t_list,
            'beta':ct['position'].T[1]/rtbpp['distance']+
            rtbpp['beta 0']+rtbpp['dot beta 0']*t_list,
            'vz':ct['velocity'].T[2]+rtbpp['w 0']}

def estimate_rtbp_parameters(obs):

    """
    Provides an initial guess for the parameters of a restricted two body problem

    :param obs: Astrometry and radial velocity
    :return: Parameters for a restricted two body problem
    """

    tb1 = {'alpha dot':numpy.diff(obs['alpha'])/numpy.diff(obs['t']),
           'beta dot':numpy.diff(obs['beta'])/numpy.diff(obs['t']),
           't':mid_array(obs['t']),
           'alpha':mid_array(obs['alpha']),
           'beta':mid_array(obs['beta']),
           'vz':mid_array(obs['vz'])}
    tb2 = {'alpha ddot':numpy.diff(tb1['alpha dot'])/numpy.diff(tb1['t']),
           'beta ddot':numpy.diff(tb1['beta dot'])/numpy.diff(tb1['t']),
           'alpha':mid_array(tb1['alpha']),
           'beta':mid_array(tb1['beta']),
           't':mid_array(tb1['t']),
           'vz dot':numpy.diff(tb1['vz'])/numpy.diff(tb1['t'])}
    def calibrate_proper_motion():
        ztrq = (tb2['beta ddot']*tb2['alpha']-
                tb2['alpha ddot']*tb2['beta'])
        aux = numpy.vstack((
            tb2['beta ddot'],
            tb2['beta ddot']*tb2['t'],
            -tb2['alpha ddot'],
            -tb2['alpha ddot']*tb2['t'])).T
        vec = numpy.einsum('n,ni', ztrq, aux)
        mat = numpy.einsum('ni,nj',aux,aux)
        return numpy.dot(numpy.linalg.inv(mat),vec)
    proper_motion = calibrate_proper_motion()
    def calibrate_z_component():
        aux = numpy.vstack((
            numpy.ones_like(tb1['beta']),
            -(tb1['alpha dot']-proper_motion[1]),
            -(tb1['beta dot']-proper_motion[3]))).T
        vec = numpy.einsum('n,ni', tb1['vz'], aux)
        mat = numpy.einsum('ni,nj', aux, aux)
        temp = numpy.dot(numpy.linalg.inv(mat),vec)
        z_list = -(temp[1]*(obs['alpha']-
                            proper_motion[0]-
                            proper_motion[1]*obs['t'])+
                   temp[2]*(obs['beta']-
                            proper_motion[2]-
                            proper_motion[3]*obs['t']))
        return temp[0],z_list
    w_0, z_list = calibrate_z_component()
    tb1['z'] = mid_array(z_list)
    tb2['z'] = mid_array(tb1['z'])
    tb2['delta alpha'] = (tb2['alpha'] -
                          proper_motion[0] -
                          proper_motion[1]*tb2['t'])
    tb2['delta beta'] = (tb2['beta']-
                         proper_motion[2]-
                         proper_motion[3]*tb2['t'])
    def calibrate_hodograph():
        aux = numpy.vstack((
            (tb1['alpha dot']-proper_motion[1])**2+
            (tb1['beta dot']-proper_motion[3])**2,
            tb1['alpha dot']-proper_motion[1],
            tb1['beta dot']-proper_motion[3],
            tb1['vz']-w_0,
            numpy.ones_like(tb1['vz']))).T
        mat = numpy.einsum('ni,nj',
                           aux,
                           aux)
        vec = numpy.einsum('n,ni',
                           (tb1['vz']-w_0)**2,
                           aux)
        return -numpy.dot(
            numpy.linalg.inv(mat),vec)
    hodograph_data = calibrate_hodograph()

    return {'alpha 0':proper_motion[0],
            'dot alpha 0':proper_motion[1],
            'beta 0':proper_motion[2],
            'dot beta 0':proper_motion[3],
            'w 0':w_0,
            'distance':numpy.sqrt(hodograph_data[0])}

    
