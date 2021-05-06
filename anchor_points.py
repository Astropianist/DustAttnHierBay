import numpy as np
import theano.tensor as tt

def anchor(x,y,zin,xfirst=None,xlast=None,xdegree=2,\
                   yfirst=None,ylast=None,ydegree=2):
    """ builds an interpolator with (xdegree,ydegree) degrees of freedom
        if (xdegree+1) * (ydegree +1) points are input, this is exact

        translates from the values of some function evaluated on an even grid between (xfirst,xlast) and 
        (yfirst,ylast). `zin` are the values of that function. 

        it then will evaluate that polynomial at locations x,y, and return that evaluation ('theta')
    """

    # evaluate for different dimensionality
    # the simple case
    if (x.ndim == y.ndim == 1):
        ngal = y.shape
        xtile = x
        ytile = y
    # the postprocessing grid
    elif (x.ndim == y.ndim == 2):
        ngal,nsamp = y.shape
        xtile = x.flatten()
        ytile = y.flatten()
    # the sampling case
    elif (x.ndim == 1) & (y.ndim == 2):
        ngal,nsamp = y.shape
        xtile = np.repeat(x,nsamp)
        ytile = y.flatten()

    theta = np.zeros(ngal)
    if (xdegree == ydegree == 0):
        return zin
    else:

        # infer coefficients
        xedge = np.linspace(xfirst,xlast,xdegree+1)
        yedge = np.linspace(yfirst,ylast,ydegree+1)
        coeffs = polyfit2d(xedge, yedge, zin, kx=xdegree, ky=ydegree)[0]

        # evaluate polynomial
        coeffs = coeffs.reshape(xdegree+1,ydegree+1)   # rearrange for np.polynomial
        theta = np.polynomial.polynomial.polyval2d(xtile, ytile,coeffs)

    # format outputs if necessary
    if (y.ndim > 1):
        theta = theta.reshape(ngal,nsamp)

    return theta

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    xx, yy = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, xx.size))
    # print(a.shape)

    # for each coefficient produce array x^i, y^j
    for index, (i, j) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(xx)
        else:
            arr = coeffs[i, j] * xx**i * yy**j
        # print(index,arr,arr.flatten())
        a[index] = arr.flatten()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)[0]

def get_a_polynd(xx):
    ''' Same as polyfitnd but just returning array that depends on grid points that goes into the least squares calculation '''
    a = np.zeros((xx[0].size,xx[0].size))
    for index, i in enumerate(np.ndindex(xx[0].shape)):
        # if sum(i)>order:
        #     arr = np.zeros_like(xx[0])
        # else:
        arr = 1
        for j,ind in enumerate(i):
            arr *= xx[j]**ind
        a[index] = arr.flatten()

    return a

def polyfitnd(xx,z):
    '''
    N-dimensional polynomial fitting by least squares.
    Fits the functional form f(x-vector) = z.

    Parameters
    ----------
    xx: np.ndarray, 3d
        Meshgrid of all variables being considered; each variable should have degree+1 points in the grid, where degree is the desired degree (same for all variables)
    z: np.ndarray, 1d
        Polynomial value at each point of the grid, flattened out
    order: int or None, default is 6
        Coefficients up to a maximum of sum(degree_given_variable) <= order are considered.

    Returns
    -------
    soln: np.ndarray (1-D if z is 1-D; 2-D if z is 2-D)
        Array of polynomial coefficients.
    '''
    a = np.zeros((xx[0].size,xx[0].size))
    for index, i in enumerate(np.ndindex(xx[0].shape)):
        # if sum(i)>order:
        #     arr = np.zeros_like(xx[0])
        # else:
        arr = 1
        for j,ind in enumerate(i):
            arr *= xx[j]**ind
        a[index] = arr.flatten()

    return np.linalg.lstsq(a.T,z,rcond=None)[0]

def calc_poly(inputarr,coefs,degree):
    ''' Calculate N-D polynomial at a set of points based on a coefficient array 
    
    Parameters
    ----------
    Inputarr: numpy 2-D or 3-D array
        Vector/matrix/tensor of new points; different variables are along the outermost axis (e.g., rows for 2-D case)
    coefs: Numpy 1-D Array
        Array of coefs that you can get from polyfitnd
    degree: Degree of polynomial in each variable (not order)
    '''
    assert (degree+1)**len(inputarr)==coefs.size
    val = 0
    sh = tuple(np.repeat(degree+1,len(inputarr)))
    for index, i in enumerate(np.ndindex(sh)):
        term = 1
        for j,ind in enumerate(i):
            term *= inputarr[j]**ind
        val+=coefs[index]*term
    return val

def calc_poly_tt(inputarr,degree):
    '''Same as calc_poly but just returns the array of (monomial) terms whose tensor dot product with the coefficient array will result in the desired polynomial'''
    sh = tuple(np.repeat(degree+1,len(inputarr)))
    inputsh = inputarr[0].shape
    shlist = list(inputsh)
    shlist.insert(0,np.prod(sh))
    term = np.ones(tuple(shlist))

    for index, i in enumerate(np.ndindex(sh)):
        for j,ind in enumerate(i):
            term[index] *= inputarr[j]**ind
    return term