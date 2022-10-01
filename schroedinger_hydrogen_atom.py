"""
/***********************************************************************************************************************/                                                                            					   			 							         		
/* Purpose: The present work centers its focus on the quantum model of the Hydrogen atom, 
            it was calculated by solving the 3D-Schrödinger’s equation. By using espherical
            coordinates and a few assumpions such as Coulombian potential and the possibility
            of performing variable separation on the wave function, it was possible to solve
            this equation, making it possible for us to study orbitals, energies and
            probability distributions of electron position. This model is quite useful 
            regarding Hydrogen and hydrogenoid’s behaviour understanding.
/***********************************************************************************************************************/
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

## 1st STEP: We represent the polar part solution, the azimuthal part solution and
##           the shape of an orbital for given values of quantum numbers (l,m)

l = 3
m = 1


#First we calculate the Legendre polynomials by the Rodrigues formula

#recursive_derivative
def recursive_deriv(f,k):
    def compute(x, dx=0.01):
        return 0.5*(f(x+dx) - f(x-dx))/dx
    if(k==0):
        return f
    if(k==1):
        return compute
    else:
        return recursive_deriv(compute,k-1)
    

def legendre(x,l):
    def f(x):
        a = (x**2-1)**l
        return a
    df = recursive_deriv(f,l)
    p_l = 1.0/(2**l*np.math.factorial(l))*df(x)
    return p_l


#Now we calculate the associated Legendre function:
    
def legendre_asoc(x,l,m):
    def f(x):
        a = legendre(x,l)
        return a
    df = recursive_deriv(f,np.abs(m))
    p_l = (1-x**2)**(np.abs(m)/2)*df(x)
    return p_l

#theta = np.arange(0, np.pi, 0.0001)

#We build the spherical harmonics

def normalizacion(l,m):
    a = (-1)**m
    s = (2*1+1)*np.math.factorial(1-m)
    t = 4*np.pi*np.math.factorial(1+m)
    b = np.sqrt(s/t)
    return a+b

def armonicos (th,ph,l,n):
    norma=normalizacion(1,m)
    f_th = np.exp(1j*m*th)
    f_ph = legendre_asoc(np.cos(ph),l,m)
    Y = norma*f_th*f_ph
    return Y

#We are going to calculate the value of the angular wave function for all angles
theta, phi = np.linspace(0,np.pi,200), np.linspace(0,2*np.pi,40)
THETA, PHI = np.meshgrid(theta,phi)
R = np.abs(armonicos(PHI,THETA,l,m))


#We change to Cartesian coordinates
X = R*np.sin(THETA) * np.cos(PHI)
Y = R*np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)


##We represent the solution for the angular part of the equation
# polar part
phi = np.linspace(0, 2*np.pi,100)

fpol = np.exp(0. + (phi*m*1j))
impol = np.imag(fpol)
repol = np.real(fpol)
abspol = abs(fpol)

ax = plt.subplot(221, projection='polar')

escalapi = (impol-np.min(impol))/np.ptp(impol)
colpi = plt.cm.coolwarm(escalapi)

plt.scatter(phi, np.abs(impol), c=colpi)
ax.grid(True)
ax.set_title("Solutions of polar axis (Im)", va="bottom")

# real part
ax = plt.subplot(222, projection='polar')

escalapr = (repol - np.min(repol))/np.ptp(repol)
colre = plt.cm.coolwarm(escalapr)

plt.scatter(phi, np.abs(repol), c=colre)
ax.grid(True)
ax.set_title("Solutions of polar axis (Re)", va="bottom")

# probability

ax = plt.subplot(223, projection='polar')

escalabs = (abspol-np.min(abspol))/np.ptp(abspol)
colabs = plt.cm.coolwarm(escalabs)

plt.scatter(phi, np.abs(abspol), c=colabs)
ax.grid(True)
ax.set_title("Solutions of polar axis (Prob)", va="bottom")



##We represent the solution for the azimuthal part of the equation

fleg = legendre_asoc(np.cos(theta) , l , m)
az = abs(fleg)

ax = plt.subplot(224, projection='polar')

escalaaz = (fleg-np.min(fleg))/np.ptp(fleg)
colaz = plt.cm.coolwarm(escalaaz)

plt.scatter(theta, np.abs(az), c=colaz)
ax.grid(True)
ax.set_title("Solution of azimuthal angular part", va="bottom")


##We represent the shape of the orbital
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("Shape of orbital from representation of spherical harmonics")
cmap = plt.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=R.min(), vmax=R.max())
plot = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=cmap(norm(R)), linewidth=0, antialiased=False, alpha=.4)

plt.show()


## 2nd STEP: We obtain the solution for the radial part of Schrödinger equation

n=5
l=3
m=0


#We calculate the Laguerre polynomial

#recursive_derivative
def recursive_deriv(f,k):
    def compute(x, dx=0.3):
        return 0.5*(f(x+dx) - f(x-dx))/dx
    if(k==0):
        return f
    if(k==1):
        return compute
    else:
        return recursive_deriv(compute,k-1)


def laguerre(x,q):
    def f(x):
        a = (np.exp(-x))*x**q
        return a
    df = recursive_deriv(f,q)
    p_l = (np.exp(x))*df(x)
    return p_l

#Now we calculate the associated Legendre function:
    
def laguerre_asoc(x,p,pq):
    def f(x):
        a = laguerre(x,pq+p)
        return a
    df = recursive_deriv(f,p)
    p_l = ((-1)**p)*df(x)
    return p_l

#now we calculate the radial function

def fun_R(n,l,m,r):
    def f(x):
        a=laguerre_asoc(x,2*l+1,n-l-1)
        return a
    a=np.math.factorial(n-l-1)
    b=2*n*(np.math.factorial(n+l))**3
    c=(2/n)**3
    funcion = np.sqrt(c*a/b)*np.exp(-r/n)*(2*r/n)**(l+1)
    return funcion*f(2*r/n)

r = np.arange(0, 8*n**1.5, 0.1)

#we apply the solution of the radial function
f_r = fun_R(n,l,m,r)

#This is the function that we are going to represent.

#We make the representation in Cartesian coordinates
ax = plt.subplot(111)

#We scale the color range
scaled_fr = (f_r - f_r.min()) / f_r.ptp()
colors = plt.cm.coolwarm(scaled_fr)
#We represent
plt.scatter(r,f_r, c=colors)
ax.grid(True)
ax.set_title("Solutions of radial axis", va='bottom')


plt.show()


## 3rd STEP: To finish, we put everything we have done so far into the slice 
##           representation of the amplitude of the wavefunction using density plots

r = np.arange(0, 10*n, 0.1)
th = np.arange(0, 2.1*np.pi, 0.1)
R,TH = np.meshgrid(r, th)

VAL = np.abs(fun_R(n,l,m,R)*armonicos(0,TH,l,m))

X = R*np.cos(TH)
Y = R*np.sin(TH)

#Set colour interpolation and colour map
colorinterpolation = 50
colourMap = plt.cm.inferno

fig = plt.figure()
ax = fig.add_subplot(111)

#Configure the contour
CTOUR = plt.contourf(Y, X, VAL, colorinterpolation, cmap=colourMap)
CTOUR
plt.colorbar(CTOUR)
plt.title("Hydrogen atom")

plt.show()