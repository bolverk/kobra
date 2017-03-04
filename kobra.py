import numpy

def generate_observational_data(rtbpp, t_list):

    """
    Generates observational data

    :param rtbpp: Restricted two body problem parameters
    :param t_list: Time list
    :return: Observational data (astrometry and radial velocity)
    """

    from brute_force import generate_complete_trajectory

    trj = generate_complete_trajectory(rtbpp,t_list)

    return {'t':t_list,
            'alpha':(trj['position'].T[0]/rtbpp['distance']+
                     rtbpp['alpha 0']+rtbpp['dot alpha 0']*t_list),
            'beta':(trj['position'].T[1]/rtbpp['distance']+
                    rtbpp['beta 0']+rtbpp['dot beta 0']*t_list),
            'vz':trj['velocity'].T[2]+rtbpp['w 0']}

def guess_proper_motion(obs):

    """
    Provides an initial guess for the proper motion parameters
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
    return {'alpha 0':res[0],
            'beta 0':res[1],
            'dot alpha 0':res[2],
            'dot beta 0':res[3],
            'lz/d**2':res[4]}

def guess_angular_momentum_ratios(obs):

    """
    Provides an initial guess for the ratios between the components of the angular momentum vector
    """

    from scipy.interpolate import UnivariateSpline

    proper_motion = guess_proper_motion(obs)
    tb1 = {field:obs[field][2:-2] for field in obs}
    for comp in ['alpha','beta']:
        spl = UnivariateSpline(obs['t'], obs[comp],k=5)
        spl.set_smoothing_factor(0)
        der = spl.derivative()
        tb1['dot '+comp] = der(tb1['t'])
    aux = numpy.vstack((
        numpy.ones_like(tb1['beta']),
        -(tb1['dot alpha']-proper_motion['dot alpha 0']),
        -(tb1['dot beta']-proper_motion['dot beta 0']))).T
    vec = numpy.einsum('n,ni',tb1['vz'],aux)
    mat = numpy.einsum('ni,nj', aux, aux)
    res = numpy.linalg.solve(mat,vec)
    return {'w 0':res[0], 'd*lx/lz': res[1], 'd*ly/lz':res[2]}

def guess_hodograph(obs):

    from scipy.interpolate import UnivariateSpline

    tb1 = {field:obs[field][2:-2] for field in obs}
    for comp in ['alpha','beta']:
        spl = UnivariateSpline(obs['t'], obs[comp], k=5)
        spl.set_smoothing_factor(0)
        der = spl.derivative()
        tb1['dot '+comp] = der(tb1['t'])

    temp = guess_angular_momentum_ratios(obs)
    w_0 = temp['w 0']
    proper_motion = guess_proper_motion(obs)
    aux = numpy.vstack((
        (tb1['dot alpha']-proper_motion['dot alpha 0'])**2+
        (tb1['dot beta']-proper_motion['dot beta 0'])**2,
        tb1['dot alpha']-proper_motion['dot alpha 0'],
        tb1['dot beta']-proper_motion['dot beta 0'],
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
        [lz_d2*l_ratios['d*lx/lz']*res['distance'],
         lz_d2*l_ratios['d*ly/lz']*res['distance'],
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
    res['semilatus rectum'] = numpy.linalg.norm(
        res['angular momentum'])**2/res['mu']
    return res

def calc_vector_angle(vec1, vec2, pvt):

    from math import copysign

    cosq = numpy.dot(vec1,vec2)
    v1cv2 = numpy.cross(vec1, vec2)
    sinq = copysign(
        numpy.linalg.norm(v1cv2),
        numpy.dot(pvt,v1cv2))

    return numpy.arctan2(sinq,cosq)

def regularise_periapse_time(raw, period):

    res = raw
    if res<0:
        res += period
    if res>period:
        res -= period
    return res

def guess_periapse_time(obs,
                        propoer_motion,
                        hodograph_data):

    from brute_force import mean_anomaly_from_true
    from brute_force import convert_mean_anomaly2time

    x_list = hodograph_data['distance']*(
        obs['alpha']-
        propoer_motion['alpha 0']-
        propoer_motion['dot alpha 0']*obs['t'])
    y_list = hodograph_data['distance']*(
        obs['beta']-
        propoer_motion['beta 0']-
        propoer_motion['dot beta 0']*obs['t'])
    z_list = -((
        x_list*hodograph_data['angular momentum'][0]+
        y_list*hodograph_data['angular momentum'][1])/
               hodograph_data['angular momentum'][2])
    q_list = numpy.array(
        [calc_vector_angle(
            [x,y,z],
            hodograph_data['eccentricity vector'],
            -hodograph_data['angular momentum'])
         for x,y,z in zip(x_list, y_list, z_list)])
    m_list = numpy.array(
        [mean_anomaly_from_true(hodograph_data['eccentricity'],q)
         for q in q_list])
    t0_array = obs['t']-numpy.array(
        [convert_mean_anomaly2time(m,{
            'eccentricity':hodograph_data['eccentricity'],
            'semilatus rectum':hodograph_data['semilatus rectum'],
            'GM':hodograph_data['mu'],
            'periapse time':0
        })
         for m in m_list])
    mean_motion = 1.0/numpy.sqrt(
        (1-hodograph_data['eccentricity']**2)**3*
        hodograph_data['mu']/hodograph_data['semilatus rectum']**3)
    period = 2*numpy.pi*mean_motion
    t0_array = numpy.array([
        regularise_periapse_time(t, period) for t in t0_array])
    return numpy.average(t0_array)

def merge_dictionaries(dic_list):

    res = {}
    for dic in dic_list:
        for field in dic:
            res[field] = dic[field]
    return res

def estimate_rtbp_parameters(obs):

    """
    Provides an initial guess for the parameters of a restricted two body problem

    :param obs: Astrometry and radial velocity
    :return: Parameters for a restricted two body problem
    """

    res = {}
    pmp = guess_proper_motion(obs)
    for field in ['alpha 0','beta 0','dot alpha 0','dot beta 0']:
        res[field] = pmp[field]
    l_ratios = guess_angular_momentum_ratios(obs)
    res['w 0'] = l_ratios['w 0']
    hodograph_data = hodograph2physical_params(
        guess_hodograph(obs),
        pmp['lz/d**2'],
        l_ratios)
    for field in ['eccentricity','distance','pivot','semilatus rectum']:
        res[field] = hodograph_data[field]
    res['GM'] = hodograph_data['mu']
    res['periapse time'] = guess_periapse_time(obs,
                                               pmp,
                                               hodograph_data)
    return res
