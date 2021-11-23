import numpy as np
from matplotlib import pyplot as plt

### Function generateTargetFunction(order, randCoeffs=True, **kwargs)
## Generates a polynomial target function based on the parameters 
## passed to it
##  
##  Parameters:
##      order: [integer] The polynomial order of the target function
##      numSamps (10): [integer] The number of samples to generate from
##                                 the target function
##      randCoeffs (True): [Boolean] Whether the coefficients should
##                                   be generated randomly or not. 
##                                   If set to false, the coefficients 
##                                   parameter must be a list of length 
##                                   matching the order parameter, 
##                                   containing real numbers to be used 
##                                   as the coefficients of the polynomial
##      coefficients: [list] Only used when the randCoeffs parameter is
##                           False. Allows manually setting the coefficients
##                           of the target function. Must be a list of real
##                           numbers
##      x_dist ('normal'): [string] A string specifying the sampling 
##                                  distribution of the x_values. 
##                                  Several other parameters will
##                                  be used or ignored depending on
##                                  this parameter (specifically 'x_mean', 
##                                  'x_std' will be used if x_dist='norm', 
##                                  and 'x_ub', 'x_lb' will be used
##                                  if x_dist='uniform')
##                                  Options are {'normal', 'uniform'}
##      x_mean (0): [int/float] Mean of the distribution to be used for 
##                            generating the samples of x (the target values
##                            input to the target function). Will only be used
##                            if x_dist='normal'
##      x_std (1): [int/float] Standard deviation of the distribution to
##                             be used for generating the samples of x 
##                             (see "x_mean" for better description of x).
##                             Will only be used if x_dist='normal'
##      x_lb (0): [int/float] Lower bound of the x-values to be generated
##                            from the uniform distribution. Will only
##                            be used if x_dist='uniform'
##      x_ub (10): [int/float] Upper bound of the x-values to be generated
##                             from the uniform distribution. Will only
##                             be used if x_dist='uniform'
##      noise_mean (0): [int/float] Mean of the noise term (epsilon)
##      noise_std (1): [int/float] Standard deviation of the noise term (epsilon).
##                                 If this is set to be less than or equal 
##                                 to 0, then no noise will be added to the 
##                                 target function
##                                  
##  Outputs: returnData
##      returnData: [dict] Dictionary with keys ('x_data', 'y_data', 'coeffs')
##              returnData['x_data']: [numpy array {numSamps by 1}] X values generated from
##                                                                  a normal distribution plugged
##                                                                  into the target function
##              returnData['y_data']: [numy array {numSamps by 1}] Y values associated with the
##                                                                 x values from the target function
##              returnData['coeffs']: [numpy array {order by 1}] The coefficients of the target function
def generateTargetFunction(order, numSamps=10, randCoeffs=True, **kwargs):

    # This is just to make this function line up with scikit learns polynomial fit
    # function, so that a 0th order function is just a constant, 
    # a 1st order function is an affine function, and a 2nd order
    # function is a function with an intercept, a linear component
    # and a quadratic term
    order = order + 1

    returnData = dict()

    if(randCoeffs == False):
        # Need to check if the coefficients variable is set
        # and then if the coefficients variable
        # is the right length
        if(not 'coefficients' in kwargs.keys()):
            raise Exception('"randCoeffs" set to False, but "coefficients" parameter not set')
        elif(not len(kwargs['coefficients']) == order):
            raise Exception('"coefficients" parameter does not have length equal to "order"')
        elif(not all(isinstance(x, (int, float) ) for x in  kwargs['coefficients']) ):
            raise Exception('"coefficients" parameter contains non-numerics')
        
        # If we got this far, that means we have the coefficients given to us already
        coeffs = np.array(kwargs['coefficients'])

    else:
        # randomly generating the coefficients from a normal(0,1) dist.
        coeffs = np.random.normal(loc=0, scale=1, size=order)
    
    
    ## Defaulting parameters


    # Checking what x_dist is set to and then defaulting parameters appropriately
    if(not 'x_dist' in kwargs.keys()):
        x_dist = 'normal'
    else:
        x_dist = kwargs['x_dist']
    
    # Checking the x_dist parameter and defaulting parameters based on this
    if(x_dist == 'normal'):
        if(not 'x_mean' in kwargs.keys()):
            x_mean = 0
        else:
            x_mean = kwargs['x_mean']
        if(not 'x_std' in kwargs.keys()):
            x_std = 1
        else:
            x_std = kwargs['x_std']
    elif(x_dist == 'uniform'):
        if(not 'x_lb' in kwargs.keys()):
            x_lb = 0
        else:
            x_lb = kwargs['x_lb']
        
        if(not 'x_ub' in kwargs.keys()):
            x_ub = 10
        else: 
            x_ub = kwargs['x_ub']
    # Only get here if the x_dist parameter was set but wasn't one of the two allowable options
    else:
        if(not isinstance(kwargs['x_dist'], str)):
            raise Exception('x_dist was set to a non-string value')
        else:

            raise Exception('Parameter passed in for x_dist was: ' + kwargs['x_dist'] + '. Which is not a viable distribution, it should be one of [\'normal\', \'uniform\']')

    if(not 'noise_mean' in kwargs.keys()):
        noise_mean = 0
    else:
        noise_mean = kwargs['noise_mean']

    if(not 'noise_std' in kwargs.keys()):
        noise_std = 1
    else:
        noise_std = kwargs['noise_std']

    if(x_dist == 'normal'):
        rng = np.random.default_rng()
        # Generating the x-values to go with the samples from a normal dist.
        x_values = rng.normal(loc=x_mean, scale=x_std, size=numSamps).reshape(numSamps, 1)
    elif(x_dist == 'uniform'):
        rng = np.random.default_rng()
        # Generating the x-values to go with the samples from a continuous uniform dist.
        x_values = rng.uniform(low=x_lb, high=x_ub, size=numSamps).reshape(numSamps, 1)
    
    # Setting up the y_values and the noise term
    y_values = np.zeros((numSamps, 1))


    if(noise_std <= 0):
        noiseValues = np.zeros((numSamps, 1))
    else:
        noiseValues = np.random.normal(loc=noise_mean, scale=noise_std, size=numSamps).reshape(numSamps, 1)

    y_values = y_values + noiseValues

    # Looping over for each of the elements of the coefficients, to generate the samples from that order
    currentOrder = 0
    for i in coeffs:
        y_values = y_values + ((x_values**currentOrder) * coeffs[currentOrder]).reshape(numSamps, 1)
        currentOrder = currentOrder + 1
    
    returnData['x_values'] = x_values
    returnData['y_values'] = y_values
    returnData['coeffs'] = coeffs
    
    return(returnData)
    
## Function plotTargetFunction(x_values, y_values, targFuncCoeffs)
## Plots a polynomial target function alongside its samples
##
##  Parameters: 
##      x_values: [list] List of the x values from the target function
##      y_values: [list] List of the y values from the target function  
##                 i.e. (f(x) = y = coeffs_0 + coeffs_1 x + coeffs_2 x^2 + ...)
##      coeffs: [list] The coefficients of the target function to be plotted
def plotTargetFunctionAndSamples(x_values, y_values, targFuncCoeffs):
    # Setting up parameters to be used for plotting
    startingXValue = np.min(x_values) - 1
    endingXValue = np.max(x_values) + 1

    targFuncPlotXVals = np.arange(startingXValue, endingXValue, step=0.01)

    targFuncPlotYVals = np.zeros((len(targFuncPlotXVals)))

    # Creating the y-values associated with the x-values by plugging into the function
    curOrder = 0
    for curCoeff in targFuncCoeffs:
        targFuncPlotYVals = targFuncPlotYVals + (curCoeff * (targFuncPlotXVals**curOrder))
        curOrder = curOrder + 1

    # Creating the plots
    plt.plot(x_values, y_values, '.')
    plt.plot(targFuncPlotXVals, targFuncPlotYVals, 'r')
    plt.xlim(left=startingXValue+0.5, right=endingXValue-0.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['samples', 'target function'], loc='upper center', bbox_to_anchor=(0.5,1.1), fancybox=True, ncol=2)

    plt.show()

## Function generatePlottableValues(polyCoeffs, x_lowerBound=0, x_upperBound=1, stepSize=0.01)
## Generates x and y values appropriate for
## plotting the polynomial function whos coefficients 
## are passed to it
##
##  Parameters: 
##      polyCoeffs: [list] List of numeric values corresponding to the
##                         coefficients of the polynomial to be plotted
##      x_lowerBound (0): [int/float] Lower bound of the x values to
##                                    generate (i.e. farthest left x-value
##                                    that you want plotted)
##      x_upperBound (1): [int/float] Upper bound of the x values to 
##                                    generate (i.e. farthest right x-value
##                                    that you want plotted)
##      stepSize (0.01): [int/float] The size of the steps to make between
##                                   the lowerbound for x and the upperbound.
##                                   the number of x & y values generated will 
##                                   be: (x_upperBound - x_lowerBound) / stepSize
##
##  Outputs: (plottableXValues, plottableYValues)
##      plottableXValues: [list] List of numeric values that are the x-values
##                               that are meant to be plotted
##      plottableYValues: [list] List of numeric values that are the y-values
##                               that are meant to be plotted (based on the 
##                               x-values and polynomial coefficients)
def generatePlottableValues(polyCoeffs, x_lowerBound=0, x_upperBound=1, stepSize=0.01):
    plottableXValues = np.arange(x_lowerBound, x_upperBound, step=stepSize)
    plottableYValues = np.zeros((len(plottableXValues)))

    curOrder = 0
    for curCoeff in polyCoeffs:
        plottableYValues = plottableYValues + (curCoeff * (plottableXValues**curOrder))
        curOrder = curOrder + 1
    return(plottableXValues, plottableYValues)


if __name__ == '__main__':
    # test = generateTargetFunction(order=5, randCoeffs=False, coefficients=[1, 0, -2, 1, 0], noise_std=1, 
    #     numSamps=10, x_mean=0)
    test = generateTargetFunction(order=4, randCoeffs=True, noise_std=1, numSamps=10, x_mean=0)
    plotTargetFunctionAndSamples(test['x_values'], test['y_values'], test['coeffs'])
