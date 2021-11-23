import numpy as np
from matplotlib import pyplot as plt

### Function generateTargetFunction(order, randCoeffs=True, **kwargs)
## Generates a polynomial target function based on the parameters 
## passed to it
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
##      coefficients: [list] only used when the randCoeffs parameter is
##                           False. Allows manually setting the coefficients
##                           of the target function. Must be a list of real
##                           numbers
##      x_mean (0): [int/float] mean of the distribution to be used for 
##                            generating the samples of x (the target values
##                            input to the target function)
##      x_std (1): [int/float] standard deviation of the distribution to
##                             be used for generating the samples of x 
##                             (see "x_mean" for better description of x)
##      noise_mean (0): [int/float] mean of the noise term (epsilon)
##      noise_std (1): [int/float] std of the noise term (epsilon).
##                                 If this is set to be less than or equal 
##                                 to 0, then no noise will be added to the 
##                                 target function
##                                  
##  Outputs:
##      returnData: [dict] dictionary with keys ('x_data', 'y_data', 'coeffs')
##              returnData['x_data']: [numpy array {numSamps by 1}] x values generated from
##                                                                  a normal distribution plugged
##                                                                  into the target function
##              returnData['y_data']: [numy array {numSamps by 1}] y values associated with the
##                                                                 x values from the target function
##              returnData['coeffs']: [numpy array {order by 1}] the coefficients of the target function
def generateTargetFunction(order, numSamps=10, randCoeffs=True, **kwargs):

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
    
    
    # Defaulting parameters
    if(not 'x_mean' in kwargs.keys()):
        x_mean = 0
    else:
        x_mean = kwargs['x_mean']
    
    if(not 'x_std' in kwargs.keys()):
        x_std = 1
    else:
        x_std = kwargs['x_std']

    if(not 'noise_mean' in kwargs.keys()):
        noise_mean = 0
    else:
        noise_mean = kwargs['noise_mean']

    if(not 'noise_std' in kwargs.keys()):
        noise_std = 1
    else:
        noise_std = kwargs['noise_std']


    # Generating the x-values to go with the samples from a normal dist.
    x_values = np.random.normal(loc=x_mean, scale=x_std, size=numSamps)
    
    
    # Setting up the y_values and the noise term
    y_values = np.zeros((numSamps))


    if(noise_std <= 0):
        noiseValues = np.zeros((numSamps))
    else:
        noiseValues = np.random.normal(loc=noise_mean, scale=noise_std, size=numSamps)

    y_values = y_values + noiseValues

    # Looping over for each of the elements of the coefficients, to generate the samples from that order
    currentOrder = 0
    for i in coeffs:
        y_values = y_values + ((x_values**currentOrder) * coeffs[currentOrder])
        currentOrder = currentOrder + 1
    
    returnData['x_values'] = x_values
    returnData['y_values'] = y_values
    returnData['coeffs'] = coeffs
    
    return(returnData)
    
## Function plotTargetFunction(x_values, y_values, targFuncCoeffs)
## Plots a polynomial target function alongside its samples
##  Parameters: 
##      x_values: [list] list of the x values from the target function
##      y_values: [list] list of the y values from the target function  
##                 i.e. (f(x) = y = coeffs_0 + coeffs_1 x + coeffs_2 x^2 + ...)
##      coeffs: [list] the coefficients of the target function to be plotted
def plotTargetFunctionAndSamples(x_values, y_values, targFuncCoeffs):
    print('hello')
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




if __name__ == '__main__':
    test = generateTargetFunction(order=5, randCoeffs=False, coefficients=[1, 0, -2, 1, 0], noise_std=1, 
        numSamps=100, x_mean=0)
    plotTargetFunctionAndSamples(test['x_values'], test['y_values'], test['coeffs'])
