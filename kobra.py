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

def guess_proper_motion(obs):

    """
    Provides an initial guess for the proper motion parameters
    """

    """
    tb1 = {field:mid_array(obs[field]) for field in obs}
    for comp in ['alpha','beta']:
        tb1['dot '+comp] = numpy.diff(obs[comp])/numpy.diff(obs['t'])
    tb2 = {field:mid_array(tb1[field]) for field in tb1}
    for comp in ['alpha','beta']:
        tb2['ddot '+comp] = numpy.diff(tb1['dot '+comp])/numpy.diff(tb1['t'])
    """

    from scipy.interpolate import UnivariateSpline

    tb1 = {field:obs[field][2:-2] for field in obs}
    for comp in ['alpha','beta']:
        spl = UnivariateSpline(obs['t'], obs[comp],k=5)
        spl.set_smoothing_factor(0)
        der = spl.derivative()
        tb1['dot '+comp] = der(tb1['t'])
    amo = tb1['dot alpha']*tb1['beta']-tb1['dot beta']*tb1['alpha']
    aux = numpy.vstack((
        tb1['dot beta'],
        -tb1['dot alpha'],
        tb1['t']*tb1['dot beta']-tb1['beta'],
        -tb1['t']*tb1['dot alpha']+tb1['alpha'],
        numpy.ones_like(tb1['beta']))).T
    vec = numpy.einsum('n,ni', amo, aux)
    mat = numpy.einsum('ni,nj',aux,aux)
    res = -numpy.linalg.solve(mat,vec)
    res[-1] += res[0]*res[3]-res[1]*res[2]
    return res

def guess_angular_momentum_ratios(obs):

    """
    Provides an initial guess for the ratios between the components of the angular momentum vector
    """

    proper_motion = guess_proper_motion(obs)
    """
    tb1 = {field:mid_array(obs[field]) for field in obs}
    for comp in ['alpha','beta']:
        tb1['dot '+comp] = numpy.diff(obs[comp])/numpy.diff(obs['t'])
    """
    from scipy.interpolate import UnivariateSpline

    tb1 = {field:obs[field][2:-2] for field in obs}
    for comp in ['alpha','beta']:
        spl = UnivariateSpline(obs['t'], obs[comp],k=5)
        spl.set_smoothing_factor(0)
        der = spl.derivative()
        tb1['dot '+comp] = der(tb1['t'])
    aux = numpy.vstack((
        numpy.ones_like(tb1['beta']),
        -(tb1['dot alpha']-proper_motion[2]),
        -(tb1['dot beta']-proper_motion[3]))).T
    vec = numpy.einsum('n,ni',tb1['vz'],aux)
    mat = numpy.einsum('ni,nj', aux, aux)
    return numpy.linalg.solve(mat,vec)

def guess_hodograph(obs):

    from scipy.interpolate import UnivariateSpline

    tb1 = {field:obs[field][2:-2] for field in obs}
    for comp in ['alpha','beta']:
        spl = UnivariateSpline(obs['t'], obs[comp], k=5)
        spl.set_smoothing_factor(0)
        der = spl.derivative()
        tb1['dot '+comp] = der(tb1['t'])

    temp = guess_angular_momentum_ratios(obs)
    w_0 = temp[0]
    proper_motion = guess_proper_motion(obs)
    #tb1 = {field:mid_array(obs[field]) for field in obs}
    #for comp in ['alpha','beta']:
    #    tb1['dot '+comp] = numpy.diff(obs[comp])/numpy.diff(obs['t'])
    aux = numpy.vstack((
        (tb1['dot alpha']-proper_motion[2])**2+
        (tb1['dot beta']-proper_motion[3])**2,
        tb1['dot alpha']-proper_motion[2],
        tb1['dot beta']-proper_motion[3],
        tb1['vz']-w_0,
        numpy.ones_like(tb1['vz']))).T
    mat = numpy.einsum('ni,nj',
                       aux,
                       aux)
    vec = numpy.einsum('n,ni',
                       (tb1['vz']-w_0)**2,
                       aux)
    return -numpy.linalg.solve(mat,vec)

def hodograph2physical_params(hod, lz_d2, l_ratios):

    res = {}
    res['distance'] = numpy.sqrt(hod[0])
    res['angular momentum'] = numpy.array(
        [lz_d2*l_ratios[1]*res['distance'],
         lz_d2*l_ratios[2]*res['distance'],
         lz_d2*res['distance']**2])
    ams = numpy.dot(res['angular momentum'],
                    res['angular momentum'])
    lce = numpy.array(hod[1:4])
    lce[0] /= res['distance']
    lce[1] /= res['distance']    
    edotmu = -0.5*numpy.cross(res['angular momentum'],
                              lce)
    res['edotmu'] = edotmu
    res['mu'] = numpy.sqrt(
        -(ams*hod[4]-numpy.dot(res['edotmu'],res['edotmu'])))
    res['eccentricity vector'] = -res['edotmu']/res['mu']
    res['eccentricity'] = numpy.sqrt(
        numpy.dot(res['eccentricity vector'],
                  res['eccentricity vector']))
    x_1 = res['eccentricity']*numpy.array([1,0,0])
    y_1 = res['eccentricity vector']
    x_2 = numpy.sqrt(ams)*numpy.array([0,0,1])
    y_2 = res['angular momentum']
    res['pivot'] = -2*numpy.linalg.solve(
        numpy.identity(3)*(numpy.dot(x_1+y_1,x_1+y_1)+
                           numpy.dot(x_2+y_2,x_2+y_2))-
        numpy.outer(x_1+y_1,x_1+y_1)-
        numpy.outer(x_2+y_2,x_2+y_2),
        numpy.cross(y_1,x_1)+numpy.cross(y_2,x_2))
    return res

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
           'alpha dot':mid_array(tb1['alpha dot']),
           'beta dot':mid_array(tb1['beta dot']),
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
        return temp[0],temp[1],temp[2],z_list
    w_0, dlx2lz, dly2lz, z_list = calibrate_z_component()
    tb1['z'] = mid_array(z_list)
    tb2['z'] = mid_array(tb1['z'])
    tb2['delta alpha'] = (tb2['alpha'] -
                          proper_motion[0] -
                          proper_motion[1]*tb2['t'])
    tb2['delta beta'] = (tb2['beta']-
                         proper_motion[2]-
                         proper_motion[3]*tb2['t'])
    lz_d2 = numpy.average(
        (tb2['delta alpha']*(tb2['beta dot']-proper_motion[1])-
         tb2['delta beta']*(tb2['beta dot']-proper_motion[3])))
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
    distance = numpy.sqrt(hodograph_data[0])
    lz = lz_d2*distance**2
    lx = lz*dlx2lz*lz
    ly = lz*dly2lz*lz
    angular_momentum = [lx,ly,lz]

    return {'alpha 0':proper_motion[0],
            'dot alpha 0':proper_motion[1],
            'beta 0':proper_motion[2],
            'dot beta 0':proper_motion[3],
            'w 0':w_0,
            'distance':numpy.sqrt(hodograph_data[0])}

    
