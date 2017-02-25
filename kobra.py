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
           'beta':mid_array(obs['beta'])}
    tb2 = {'alpha ddot':numpy.diff(tb1['alpha dot'])/numpy.diff(tb1['t']),
           'beta ddot':numpy.diff(tb1['beta dot'])/numpy.diff(tb1['t']),
           'alpha':mid_array(tb1['alpha']),
           'beta':mid_array(tb1['beta']),
           't':mid_array(tb1['t'])}
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

    return {'alpha 0':proper_motion[0],
            'dot alpha 0':proper_motion[1],
            'beta 0':proper_motion[2],
            'dot beta 0':proper_motion[3]}

    
