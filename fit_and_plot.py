import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import warnings
def bootstrap_fit(x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper, gd):
    #Run Bootstrap one time
    n_points = len(x)
    bootstrap_indices = np.random.choice(n_points, size=n_points, replace=True)
    x_bs = x[bootstrap_indices]
    y_bs = y[bootstrap_indices]
    x_err_lower_bs = x_err_lower[bootstrap_indices] if x_err_lower is not None else None
    x_err_upper_bs = x_err_upper[bootstrap_indices] if x_err_upper is not None else None
    y_err_lower_bs = y_err_lower[bootstrap_indices] if y_err_lower is not None else None
    y_err_upper_bs = y_err_upper[bootstrap_indices] if y_err_upper is not None else None
    initial_guess=np.polyfit(x_bs,y_bs,1)
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")
        bs_result = minimize(
            gd, 
            initial_guess, 
            args=(x_bs, y_bs, x_err_lower_bs, x_err_upper_bs, y_err_lower_bs, y_err_upper_bs), 
            method='Nelder-Mead'
        )
    return bs_result.x if bs_result.success else None
def gaussian_distance(x, a, ex):
    #calculate the weighted residual
    z = np.abs(x - a) / ex
    distance = norm.cdf(z) - norm.cdf(-z)
    return distance*np.abs(x-a)
def xgd(x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper,a,b):
    #calculate the pseudo-orthogonal weighted residual between one data point and the fitted line
    if x>(y-b)/a:
        xerr= x_err_lower
    else:
        xerr= x_err_upper
    if y>(a*x+b):   
        yerr= y_err_lower
    else:
        yerr= y_err_upper
    gdx=gaussian_distance(x, (y-b)/a, xerr)
    gdy=gaussian_distance(y, a*x+b, yerr)
    return np.sqrt(gdx**2+gdy**2)
def xgd2(x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper,a,b):
    #after getting the best fit, calculate the pseudo-orthogonal weighted residual between the data and the fitted line
    xflag=1
    if x>(y-b)/a:
        xerr= x_err_lower
    else:
        xerr= x_err_upper
        xflag=-1
    yflag=1
    if y>(a*x+b):   
        yerr= y_err_lower
    else:
        yerr= y_err_upper
        yflag=-1
    gdx=xflag*gaussian_distance(x, (y-b)/a, xerr)
    gdy=yflag*gaussian_distance(y, a*x+b, yerr)
    return (gdx,gdy,xflag)

def gd(para,x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper):
    #calculate the overall pseudo-orthogonal weighted residual between the data and the fitted line
    sum=0
    a=para[0]
    b=para[1]
    for i in range(len(x)):
        sum+=xgd(x[i], y[i], x_err_lower[i], x_err_upper[i], y_err_lower[i], y_err_upper[i],a,b)
    return sum
def main(x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper,xname,yname,a,b,aerr,berr,n_bootstrap):
    #xname = 'X-axis label'
    #yname = 'Y-axis label'
    #if a and b are None, then fit, else skip fitting
    #if aerr and berr are None, then skip error calculation, else calculate error
    #if n_bootstrap is None, then skip bootstrap, else do bootstrap
    if a==None and b==None:
        initial_guess=np.polyfit(x,y,1)
        result = minimize(gd, initial_guess, args=(x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper), method='Nelder-Mead')
        a, b = result.x
    rex=x-x
    rey=y-y
    flag=x-x
    for i in range(len(rex)):
        rex[i],rey[i],flag[i]=xgd2(x[i], y[i], x_err_lower[i], x_err_upper[i], y_err_lower[i], y_err_upper[i],a,b)
    res=flag*np.sqrt(rex**2+rey**2)
    if aerr==None and berr==None:
        if n_bootstrap!=None:
            func = partial(
                bootstrap_fit,
                x=x, y=y, x_err_lower=x_err_lower, x_err_upper=x_err_upper,
                y_err_lower=y_err_lower, y_err_upper=y_err_upper, gd=gd
            )
            results=[]
            for i in range(n_bootstrap):
                print("bootstrap counts: "+str(i),end='\r')
                results.append(func())
            bootstrap_params = np.array([r for r in results if r is not None])
        else:
            bootstrap_params = None
            slope_err=-999
            Intercept_err=-999
    else:
        bootstrap_params = None
    print(f"Fitted parameters: a = {a}, b = {b}")
    if n_bootstrap!=None:
        slope=a
        Intercept=b
        if bootstrap_params is not None:
            slope_err = np.std(bootstrap_params[:, 0])
            Intercept_err = np.std(bootstrap_params[:, 1])
        else:
            slope_err = aerr
            Intercept_err = berr
        total_dis=np.std(res)
        error_dis=np.sqrt(np.std(x_err_lower)**2+np.std(x_err_upper)**2+np.std(y_err_lower)**2+np.std(y_err_upper)**2)
        intrinsic_dis=np.sqrt(total_dis**2-error_dis**2)
        textstr = '\n'.join((
            f'Slope: {slope:.2f} ± {slope_err:.2f}',
            f'Intercept: {Intercept:.2f} ± {Intercept_err:.2f}',
            f'Total scatter: {total_dis:.2f}',
            f'Error scatter: {error_dis:.2f}',
            f'Intrinsic scatter: {intrinsic_dis:.2f}'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, verticalalignment='top', bbox=props)
    plt.errorbar(x,y, xerr=[x_err_lower, x_err_upper], yerr=[y_err_lower, y_err_upper], fmt='o', label='Data',alpha=0.1)
    plt.plot(x, a*x+b,'-', color='black',label='Fitted line')
    plt.xlabel(xname,fontsize=16)##########################
    plt.ylabel(yname,fontsize=16)##########################
    plt.xticks(fontsize=16)##########################
    plt.yticks(fontsize=16)##########################
    print("parameters error:", (slope_err, Intercept_err))
    return a,b,res
if __name__ == "__main__":
    df=pd.read_csv('sample.csv')
    df=df[df['Above_KB']]
    df=df[df['Mass']>1.6]
    df=df[df['Mass']<=1.8]
    x=df['Age'].values
    RR_P=df['RR_P'].values
    x_err_lower = -df['Age_err_l'].values
    x_err_upper = df['Age_err_u'].values
    RR_P_err_lower = np.log10(RR_P)-np.log10(RR_P+df['RR_P_err_final_l'].values)
    RR_P_err_upper = np.log10(RR_P+df['RR_P_err_final_u'].values)-np.log10(RR_P)
    RR_P=np.log10(RR_P)
    feh=df['feh'].values
    feh_err_lower = -df['feh_err_l'].values
    feh_err_upper = df['feh_err_u'].values
    mass=df['Mass'].values
    mass_err_lower = -df['Mass_err_l'].values
    mass_err_upper = df['Mass_err_u'].values
    a,b,res=main(mass,RR_P, mass_err_lower, mass_err_upper, RR_P_err_lower, RR_P_err_upper,'Mass ($M_{\\odot}$)','$\\log RR_P$',None,None,0.17,0.29,100)
    plt.tight_layout()
    plt.show()
    exit()
