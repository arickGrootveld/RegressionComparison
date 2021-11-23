from generateTargetFunction import generateTargetFunction, generatePlottableValues

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import pyplot as plt

import numpy as np


## fitPolynomialRegressionLine(xVals, yVals, degree=1)
## Function that fits a polynomial regression line of
## degree 'degree' to the xVals and yVals passed to it
##
##  Parameters: 
##      xVals: [list] A list of numeric values that are the
##                    'features' to train the regression line
##                    on
##      yVals: [list] A list of numeric values that are
##                    the true values you want to predict
##                    or estimate
##      degree (1): [int] The degree of the polynomial that
##                        will be fit to the data
##
##  Outputs: (regressionCoeffs, regressionObject)
##      regressionCoeffs: [list] A list of numeric values that 
##                               are the coefficients of the 
##                               polynomial regression line
##      polyRegressionObject: [sklearn.linear_model] The scikit-learn linear regression object that serves as our polynomial regression model. 
##                                               See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
##                                               for more details
def fitPolynomialRegressionLine(xVals, yVals, degree=1):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(xVals)

    # Create the polynomial regression object
    polyRegressionObject = LinearRegression()
    polyRegressionObject.fit(x_poly, yVals)

    # Get the coefficients of the polynomial regression
    regressionCoeffs = polyRegressionObject.coef_[0]
    regressionCoeffs[0] = polyRegressionObject.intercept_

    return(regressionCoeffs, polyRegressionObject)


## Function testPolynomialRegression(xVals, yVals, polyRegressionObject)
## A function to make a prediction of the yVals given the xVals, and 
## a polynomial scikit-learn regression object
##
## Parameters: 
##      xVals: [list] A list of numeric values that are the x-values
##                    to make predictions on
##      yVals: [list] A list of numeric values that correspond to the
##                    true values corresponding to the x-values
##      polyRegressionObject: [sklearn.linear_model] A scikit-learn linear regression model that is the polynomial regression function
##                                                   that we will be using to make predictions of the y-values from the x-values.
##                                                   See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
##                                                   for more details
##
## Outputs: (predictedValues, residuals, regressionMSE)
##      predictedValues: [list] A list of numeric values that are the predictions of the y-values
##                              that the polynomial regression line found from the x-values
##      residuals: [list] A list of numeric values that are the differences between the 
##                        true y-values and our predictions of them. Calculated by:
##                        residuals = (y_pred - y_true) {with y_pred being our predictions
##                        and y_true being the yVals passed to the function}
##      regressionMSE: [float] The mean squared error or our regression line and the true
##                             y-values. Calculated by: regressionMSE = 1/N sum(residuals^2)
##                             {where N is the number of y values passed to the function}
def testPolynomialRegression(xVals, yVals, polyRegressionObject):
    # Converting the x_values to the appropriate polynomial dimension before performing the regression
    poly = PolynomialFeatures(degree=len(polyRegressionObject.coef_[0]) - 1)
    polyXVals = poly.fit_transform(xVals)

    predictedValues = polyRegressionObject.predict(polyXVals)
    residuals = predictedValues - yVals

    regressionMSE = (1/len(yVals)) * np.sum(residuals**2)

    return(predictedValues, residuals, regressionMSE)


## Function plotTargFunctionAndRegressionFunction(targetFunction, regressionFuncCoefficients)
## Plots the target function and the regression function alongside each other, as well as the
## samples
##  
##  Parameters: 
##      targetFunction: [dict] A dictionary containing (at least) three key-value pairs:
##                             'x_values', 'y_values', and 'coeffs' (see output of 
##                              "generateTargetFunction" for more details)
##          targetFunction['x_values']: [list] A list of numeric values that correspond
##                                             to the x values of the target function
##          targetFunction['y_values]: [list] A list of numeric values that correspond
##                                            to the y values of the target function
##          targetFunction['coeffs']: [list] A list of numeric values that are the 
##                                           coefficients of the target function
##      regressionFuncCoefficients: [list] A list of numeric values that contains
##                                         the coefficients of the regression function
##                                         that is supposed to be plotted alongside the
##                                         target function
def plotTargFunctionAndRegressionFunction(targetFunction, regressionFuncCoefficients):
    targFunc_xVals = targetFunction['x_values']
    targFunc_yVals = targetFunction['y_values']
    targFunc_coeffs = targetFunction['coeffs']

    startingXValue = np.min(targFunc_xVals) - 1
    endingXValue = np.max(targFunc_xVals) + 1


    # Creating the target function y-values from its polynomial coefficients
    [plotXVals, targFuncPlotYVals] = generatePlottableValues(targFunc_coeffs, x_lowerBound=startingXValue, x_upperBound=endingXValue, stepSize=0.01)
    
    # Creating the regression function y-values from its polynomial coefficients
    [plotXVals, regressionFuncPlotYVals] = generatePlottableValues(regressionFuncCoefficients, x_lowerBound=startingXValue, x_upperBound=endingXValue, stepSize=0.01)

    # Plotting the target function alongside the regression line, 
    # as well as the samples from the target function

    # Plotting the samples
    plt.plot(targFunc_xVals, targFunc_yVals, '.')
    
    # Plotting the target function
    plt.plot(plotXVals, targFuncPlotYVals, 'r')

    # Plotting the regression function
    plt.plot(plotXVals, regressionFuncPlotYVals, 'b')

    # Filling out the plot
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['samples', 'target function', 'regression line'], loc='upper center', bbox_to_anchor=(0.5,1.1), fancybox=True, ncol=2)

    plt.show()

if __name__ == '__main__':
    ## Parameters of the simulation
    degreeOfTarget = 10
    degreeOfRegression = 2
    numTrainSamples = 2
    numTestSamples = 1000

    # Generate a target function and then perform regression analysis, and then report and plot the results
    targFunc_Train = generateTargetFunction(degreeOfTarget, numSamps=numTrainSamples)
    targFunc_Test = generateTargetFunction(degreeOfTarget, numSamps=numTestSamples, randCoeffs=False, coefficients=targFunc_Train['coeffs'])


    [regressionCoeffs, polyRegression] = fitPolynomialRegressionLine(targFunc_Train['x_values'], targFunc_Train['y_values'], degree=degreeOfRegression)

    (regressionTestValues, regressionResiduals, regressionMSE) = testPolynomialRegression(targFunc_Test['x_values'], targFunc_Test['y_values'], polyRegression)


    # Plotting the target function and the regression line
    plotTargFunctionAndRegressionFunction(targFunc_Test, regressionFuncCoefficients=regressionCoeffs)

    print('this is here')


