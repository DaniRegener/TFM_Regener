#!/bin/env python
'''Run a sequence of test 2D cases varying the security factor f
 Author:     Daniel Regener Roig
 Supervisor: Arnau Miró Jané
 Date:       17/08/2023
 Program developed for the master's thesis "TFM-220MUAERON- 
 Advanced methods for numerical simulations of turbulent flows"
 ESEIAAT - UPC

 Description: This program simulates the incompressible navier stokes equations in a fully
periodid 2D domain and knwon solutions. Set fmin and fmax and the program will be launched
successivelly for all the range of f in steps of 0.1. in line 1563 you can find the metrics
that are printed for each case. This program is launched inside run_some_f_test.sh'''
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Problem parameters
L    = 1    # cubic domain side size [m]
N    = 16   # number of CV per axis, do not try N != M please
M    = 16   # please do not try N!=M
fmin = 0.1  # start security factor f
fmax = 4    # end security factor f
Re   = 200  # Reynolds number
tend = 5    # end time
eigen_tstep = True      # if eigenCD timestep is desired
approximate_ROS = False # only if eigen_tstep is True and using a RK method


forced_tstep = False
hardDt = 0.15      # if fixed Dt is wanted

integration = 'LF'
# Options:
#	> AdamsBashforth (AB)
#	> LeapFrog (LF)
#	> Explicit Runge-Kutta (RK)
#	> Low Storage Explicit Runge-Kutta (LSRK) 'LSRungeKutta'
RK_method	  = 'SSP_RK3'
# Options:
#    > midpoint_RK2
#    > heun_RK3
#    > ralston_RK3
#    >  kutta_RK3
#    > SSP_RK3
#    > classic_RK4
#    > SSP_RK4
#    > 3p5q
#    > 4p7q
# 2NRK options:
#    > RK3LR
#    > RK3LT
#    > RK4s5
#    > RK4s6
#    > RK4s7
#    > RK4s12
#    > RK4s13
#    > RK4s14
#    > LDDRK2s5
#    > LDDRK4s6
approximation = 'Approximated_2'
# Options:
#    > Consistent
#    > Approximated_1 (pss=ps)
#    > Approximated_2 (different extrapolations)

# Physical data
nu        = 1/Re     # kinematic viscosity of the fluid [m/s²]
rho       = 1        # density of the fluid [kg/m³]
converged = 0

# Studied fields
F    = lambda t     :  np.exp( -8*(np.pi**2) * nu * t )                 # F factor for the velocities
ufun = lambda x,y,t :  F(t) * np.cos( 2*np.pi*x ) * np.sin( 2*np.pi*y ) # U field [m/s]
vfun = lambda x,y,t : -F(t) * np.sin( 2*np.pi*x ) * np.cos( 2*np.pi*y ) # V field [m/s]
# Studied point
iref, jref = 2, 2 # in INNER MESH coordinates


## MAIN FUNCTIONS
def create_mesh(L,N,M):
	'''
	 create_mesh(L,N) : creates a uniform square mesh for a 2D prolem
	
	 Returns a structure containing the information needed of the square
	   uniform mesh for solving a simple CFD problem. Halo size is 1. Be
	   warned that the first control volume is situated in the halo corner
	   and the zero coordinates are situated on the corner of the 
	   principal mesh.
	
			 h
	   s(3)+---+---+---+---    A: center of coordinates i.e x=0 y=0
	   c(3)|\\\|   |   |       B: CV with i=1, j=1
	   s(2)+---+---+---+--
	   c(2)|\\\|   |   |    
	   s(1)+---A---+---+--
	 y c(1)|\B\|\\\|\\\|  h
	 ^     +---+---+---+--   
	 |      c(1) ..same than y..
	  --> x
	
	 Input:
	   L: scalar, total lengh of every side of the principal mesh
	   N: scalar, number of CV in each direction
	
	 Output:
	   mesh: stucture containing information as
		   mesh.N: scalar, number of CV in each direction
		   mesh.delta: scalar, distance between neighbour points
		   mesh.c: vector, coordinates of centered CV (as the mesh is
			   square and uniform its the same for x and y)
		   mesh.s: vector, coordinates of staggered CV (as the mesh is
			   square and uniform its the same for x and y)
	
	 Example of use
	   mesh = create_mesh(1,4)
	'''
	assert N > 2
	mesh = {
		'N'     : [N,M],     # number of CV in each axis
		'delta' : [L/N,L/M], # delta in each axis
		'cx'    : np.zeros((N+2,),np.double),
		'sx'    : np.zeros((N+2,),np.double),
		'cy'    : np.zeros((M+2,),np.double),
		'sy'    : np.zeros((M+2,),np.double),
	}

	for i in range(N+2):  # for desired CVs plus halos
		mesh['cx'][i] = -mesh['delta'][0]/2. + i*mesh['delta'][0] # centered cordinates in ax axis
		mesh['sx'][i] =  mesh['delta'][0]*i                       # staggered coordinates in ax axis

	for i in range(M+2):
		mesh['cy'][i] = -mesh['delta'][1]/2. + i*mesh['delta'][1] # centered cordinates in ax axis
		mesh['sy'][i] =  mesh['delta'][1]*i                       # staggered coordinates in ax axis

	return mesh

def set_field(ufun,vfun,td,mesh):
	'''
	 set_field(mesh): Compute the values of the field at the corresponding coordinates
	 Return the value of the corresponding field at each point of the mesh

	 Input:
	   mesh: structure, refer to create_mesh.m for further information  

	 Output:
	   u: square array, values of the horizontal velocity field, the outer rows and colums are
		   not relevant
	   v: square array, values of the vertical velocity field, the outer rows and colums are
		   not relevant
	'''
	N  = mesh['N'][0]
	M  = mesh['N'][1]

	# initialization of the evaluated velocity vectors
	u = np.zeros((N+2,M+2),np.double)
	v = np.zeros((N+2,M+2),np.double)
	  
	# evaluate at nodal positions
	for i in range(1,N+1):
		for j in range(1,M+1):
			u[i,j] = ufun(mesh['sx'][i],mesh['cy'][j],td)
			v[i,j] = vfun(mesh['cx'][i],mesh['sy'][j],td)
	return u,v

def halo_update(u):
	'''
	halo_update( u ): halo update of a square 2D scalar field
	
	Returns the halo updated field considering a halo size of 1.

	Input:
	  u: square array, values of the field, the outer rows and colums are
		  not relevant

	Output:
	  updated: structure containing field with the corect values assigned
	  to the halo, corners of the halo are not relevant.

	  +---+---+---+---+---+
	  |\\\|\G\|\H\|\I\|\\\|
	  +---+---+---+---+---+
	  |\C\| A | B | C |\A\|
	  +---+---+---+---+---+
	  |\F\| D | E | F |\D\|
	  +---+---+---+---+---+
	  |\I\| G | H | I |\G\|
	  +---+---+---+---+---+
	  |\\\|\A\|\B\|\C\|\\\|
	  +---+---+---+---+---+  
	
	Example of use
	  halo_update(eye(4))
	'''
	updated = u.copy()
	N = u.shape[0]-2 # mesh size x
	M = u.shape[1]-2 # mesh size x

	# remember that in u the values are transposed compared to the figure
	for i in range(N+2):
		updated[i,0]   = u[i,M] # GHI values to the halos
		updated[i,M+1] = u[i,1] # ABC ...

	for j in range(M+2):
	   updated[0,j]   = u[N,j] # ADG ...
	   updated[N+1,j] = u[1,j] # CFI ... 

	updated[-1,-1] = u[0,0]
	updated[0,0]   = u[-1,-1]
	updated[-1,0]  = u[0,-1]
	updated[0,-1]  = u[-1,0]    

	return updated

def CFLcondition(f,nu,u,v,gamma,mesh):
	'''
	maxDt(f, nu, u, v, mesh): compute the maximum dt for stability 
	
	Input:
	  f: double, factor applied to CFL condition for ensuring stability
	  nu: double, kinematic viscosity of the fluid
	  u: square array, values of the x velocity field, the outer rows and 
		  colums are not relevant
	  v: square array, values of the y velocity field, the outer rows and 
		  colums are not relevant
	  mesh: structure containing the information of the mesh, refer to
			  create_mesh.m for further information

	Output:
	  dt: double, time for integration of the NS equations
	'''
	# start the minimums at an ARBITRARY HIGH value
	dtCu_min = 1e10
	dtCv_min = 1e10
	dtD_min  = 1e10
	
	# loop the INNER MESH
	for i in range(1,mesh['N'][0]+1):
		for j in range(1,mesh['N'][1]+1):
			dtCu = mesh['delta'][0]/abs(u[i,j]) # dt_c=delta/|u|
			dtCv = mesh['delta'][1]/abs(v[i,j]) # dt_c=delta/|v|
			dtD  = min(0.5*mesh['delta'][0]**2/nu,0.5*mesh['delta'][1]**2/nu)     # dt_d=delta^2/nu
			dtCu_min = min(dtCu_min,dtCu) # if calculated value is lower, save it
			dtCv_min = min(dtCv_min,dtCv)
			dtD_min  = min(dtD_min ,dtD)
	return f*np.min([dtCu_min,dtCv_min,dtD_min]), 0.5 # get the mini of the min and apply f

def gradient(d,mesh):
	'''
	grad(d,mesh): Compute the gradient of the pressure
	
	Input:
	  d(N+2,N+2): scalar field of pseudo-pressure
	Output:
	  gx,gy: vector field, staggered, of pressure gradient for each
	  direction
	
	Example of use
	  grad(d,mesh)
	'''
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	dx = mesh['delta'][0]
	dy = mesh['delta'][1]

	gx, gy = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	
	for i in range(1,N+1):
		for j in range(1,M+1): # Loops all the mesh
			gx[i,j] = (d[i+1,j] - d[i,j])/dx
			gy[i,j] = (d[i,j+1] - d[i,j])/dy
		
	return halo_update(gx), halo_update(gy) # update the halo

def divergence(u,v,mesh):
	'''
	diverg(u,v,mesh): Compute the divergence of up
	
	Input:
	  u,v(N+2,N+2): velocity vector field (staggered) in each direction
	  mesh: structure, refer to create_mesh.m for further information  
	Output:
	  d(N+2,N+2): scalar field (centred) of the velocity divergence
	Example of use
		  divergence(u,v,mesh)
	'''
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	dx = mesh['delta'][0]
	dy = mesh['delta'][1]
	
	d = np.zeros((N+2,M+2),np.double)

	for i in range(1,N+1):
		for j in range(1,M+1): # Loops all the mesh
			d[i,j] = dx*(v[i,j] - v[i,j-1]) + dy*(u[i,j] - u[i-1,j])
	
	return halo_update(d)

def convection_skew(mesh,u,v):
	'''
	convection(mesh,u,v): numerical convection of 2D field

	 Returns the convection for each of its components of a two
	   dimensional field in each point of the mesh as Ci = uj(dui/dxj)
	   
	 Input:
	   mesh: structure, refer to create_mesh.m for further information 
	   u: square array, values of the horizontal component of the velocity,
	   the outer rows and colums are not relevant
	   v: square array, values of the vertical component of the velocity,
	   the outer rows and colums are not relevant
	 Output:
	   Cu: convection of the horizontal component of the field
	   Cv: convection of the vertical component of the field

	Example of use
		convection_skew(mesh,[1,1;1,1],[2,2;2,2])
	'''
	# Initialization of variables
	N, M  = mesh['N'][0], mesh['N'][1]
	dx,dy = mesh['delta'][0], mesh['delta'][1]

	Cu, Cv = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	volume = dx*dy

	for i in range(1,N+1):
		for j in range(1,M+1): # Loops all the mesh
			# horizontal component
			# velocities at CV faces
			ue = 0.5*(u[i+1,j] + u[i,j]) 
			uw = 0.5*(u[i-1,j] + u[i,j])
			un = 0.5*(u[i,j+1] + u[i,j])
			us = 0.5*(u[i,j-1] + u[i,j])
			
			# perpendicular flow at CV faces
			Fe_u = dy*0.5*(u[i+1,j] + u[i,j]    )
			Fw_u = dy*0.5*(u[i,j]   + u[i-1,j]  )
			Fn_u = dx*0.5*(v[i+1,j] + v[i,j]    )
			Fs_u = dx*0.5*(v[i,j-1] + v[i+1,j-1])
			
			# convection u component
			Cu[i,j] = (ue*Fe_u - uw*Fw_u + un*Fn_u - us*Fs_u)/volume 
			
			# vertical component
			# velocites at CV faces
			ve = 0.5*(v[i,j] + v[i+1,j])
			vw = 0.5*(v[i,j] + v[i-1,j])
			vn = 0.5*(v[i,j] + v[i,j+1])
			vs = 0.5*(v[i,j] + v[i,j-1])
			
			# perpedicular flow at CV faces
			Fe_v = dy*0.5*(u[i,j+1] + u[i,j]    )
			Fw_v = dy*0.5*(u[i-1,j] + u[i-1,j+1])
			Fn_v = dx*0.5*(v[i,j+1] + v[i,j]    )
			Fs_v = dx*0.5*(v[i,j]   + v[i,j-1]  )
			
			# convection v component
			Cv[i,j] = (ve*Fe_v - vw*Fw_v + vn*Fn_v - vs*Fs_v)/volume 

	return Cu, Cv

def convection_new(mesh,u,v,alpha=0.):
	'''
	convection(mesh,u,v): numerical convection of 2D field

	 Returns the convection for each of its components of a two
	 dimensional field in each point of the mesh as Ci = uj(dui/dxj)
	
	 Implements the relationships from the following papers:
		Morinishi, Y., Lund, T.S., Vasilyev, O.V., Moin, P., 1998. 
		Fully Conservative Higher Order Finite Difference Schemes for 
		Incompressible Flow. Journal of Computational Physics 143, 90–124. 
		
		Edoh, A.K., 2022. A new kinetic-energy-preserving method based on the 
		convective rotational form. Journal of Computational Physics 454, 110971. 

	 In a nutshell, the convective term is expressed as:
	 	(u·D)u = D·(uu) - u(D·u)
	 being:
	 	C_div = D·(uu) as the divergence form of the convective term
	 	C_adv = (u·D)u as the advective form of the convective term
	
	 Then we can express the skew-symmetric convective term as
	 	C_skew = 0.5*D·(uu) + 0.5*(u·D)u = 0.5*C_div + 0.5*C_adv
	 
	 Notice that for incompressible flows D·u = 0 (a priori), thus
		C_adv = C_div -> C_skew = C_div
	 hence the divergence term is (a priori) energy preserving. However, at a discrete
	 level D·u != 0 (e.g., in the predictor-corrector step) and hence the 3 forms
	 of expressing the convective term differ. In particular, only C_skew is energy
	 preserving as the symmetries are corrected.

	 Edoh generalizes this concept through a means of a parameter
	 	C = alpha*C_adv + (1-alpha)*C_div
	 and for alpha = 0.5 the skew-symmetric formulation is recovered. Moreover, Edoh
	 proposes two additional schemes based on the EMAC idea of correcting the diagonal
	 of the discrete matrix as
	 	C_corr = dij*(C_adv - C_div) = dij*(- u(D·u))
	 and taking advantage of the properties of the divergence and advective forms. 
	 These are the div-rot and div-str methods. 
	 	C_div-str = C_div + C_corr = D·(uu) + 0.5*dij*(- u(D·u))
	 	C_div-rot = C_adv - C_corr = D·(uu) - 0.5*dij*(- u(D·u))

	 In particular, this can be formulated as a one-parameter scheme as
	 	C = alpha*C_adv + (1-alpha)*C_div + (1-2*alpha)/2*(C_adv - C_div)
	 then
	 	alpha = 0.  -> recovers the div-str scheme (preserves energy and angular momentum a priori)
	 	alpha = 0.5 -> recovers the skew-symmetric scheme (preserves energy)
	 	alpha = 1.0 -> recovers the div-rot scheme (preserves energy and helicity)
	'''
	# Initialization of variables
	N, M  = mesh['N'][0], mesh['N'][1]
	dx,dy = mesh['delta'][0], mesh['delta'][1]

	Cu, Cv = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	volume = dx*dy

	for i in range(1,N+1):
		for j in range(1,M+1): # Loops all the mesh
			# horizontal component
			# velocities at CV faces
			uc = u[i,j]
			ue = 0.5*(u[i+1,j] + u[i,j]) 
			uw = 0.5*(u[i-1,j] + u[i,j])
			un = 0.5*(u[i,j+1] + u[i,j])
			us = 0.5*(u[i,j-1] + u[i,j])
			
			# perpendicular flow at CV faces
			Fe_u = dy*0.5*(u[i+1,j] + u[i,j]    )
			Fw_u = dy*0.5*(u[i,j]   + u[i-1,j]  )
			Fn_u = dx*0.5*(v[i+1,j] + v[i,j]    )
			Fs_u = dx*0.5*(v[i,j-1] + v[i+1,j-1])

			# divergence part
			Fd_u = dy*0.5*(u[i+1,j] - u[i-1,j])
			Fd_v = dx*0.5*( (v[i+1,j] + v[i,j]) - (v[i,j-1] + v[i+1,j-1]) )

			# fluxes
			F_div = ue*Fe_u - uw*Fw_u + un*Fn_u - us*Fs_u # Divergence approximation at the convective fluxes (before dividing by the volume)
			F_adv = F_div - uc*(Fd_u+Fd_v)                # Advective approximation at the convective fluxes (before dividing by the volume)
			F_cor = -uc*(Fd_u+Fd_v)                       # Correction at the diagonal corresponding to F_adv - F_div

			# convection u component
			Cu[i,j] = (alpha*F_adv+(1-alpha)*F_div+(1-2*alpha)/2*F_cor)/volume
			
			# vertical component
			# velocites at CV faces
			vc = v[i,j]
			ve = 0.5*(v[i,j] + v[i+1,j])
			vw = 0.5*(v[i,j] + v[i-1,j])
			vn = 0.5*(v[i,j] + v[i,j+1])
			vs = 0.5*(v[i,j] + v[i,j-1])
			
			# perpedicular flow at CV faces
			Fe_v = dy*0.5*(u[i,j+1] + u[i,j]    )
			Fw_v = dy*0.5*(u[i-1,j] + u[i-1,j+1])
			Fn_v = dx*0.5*(v[i,j+1] + v[i,j]    )
			Fs_v = dx*0.5*(v[i,j]   + v[i,j-1]  )

			# divergence part
			Fd_u = dy*0.5*( (u[i,j+1] + u[i,j]) - (u[i-1,j] + u[i-1,j+1]) )
			Fd_v = dx*0.5*(v[i,j+1] - v[i,j-1])
			
			# fluxes
			F_div = ve*Fe_v - vw*Fw_v + vn*Fn_v - vs*Fs_v # Divergence approximation at the convective fluxes (before dividing by the volume)
			F_adv = F_div - uc*(Fd_u+Fd_v)                # Advective approximation at the convective fluxes (before dividing by the volume)
			F_cor = -uc*(Fd_u+Fd_v)                       # Correction at the diagonal corresponding to F_adv - F_div

			# convection v component
			Cv[i,j] = (alpha*F_adv+(1-alpha)*F_div+(1-2*alpha)/2*F_cor)/volume

	return Cu, Cv

def diffusion(mesh,u,v):
	'''
	diffusion(mesh,u,v): numerical diffusion of 2D field
	Returns the diffusion for each of its components of a two
	  dimensional field in each point of the mesh as Di = d²ui/dxj²
	
	Input:
	  mesh: structure, refer to create_mesh.m for further information 
	  u: square array, values of the horizontal component of the velocity,
	  the outer rows and colums are not relevant
	  v: square array, values of the vertical component of the velocity,
	  the outer rows and colums are not relevant
	Output:
	  Du: diffusion of the horizontal component of the field
	  Dv: diffusion of the vertical component of the field
	
	Example of use
	  diffusion(mesh,[1,1;1,1],[2,2;2,2])	
	'''
	# Initialization of variables
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	dx = mesh['delta'][0]
	dy = mesh['delta'][1]
	
	Du, Dv = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	dy_dx  = dy/dx
	volume = dx*dy

	for i in range(1,N+1):
		for j in range(1,M+1): # Loops all the mesh
			aux_u, aux_v = 0., 0.
			
			# horizontal component
			aux_u +=  (u[i+1,j] - u[i,j]  )*dy_dx # e
			aux_u += -(u[i,j]   - u[i-1,j])*dy_dx # w
			aux_u +=  (u[i,j+1] - u[i,j]  )*dy_dx # n
			aux_u += -(u[i,j]   - u[i,j-1])*dy_dx # s
			
			# diffusion u component
			Du[i,j] = aux_u/volume
			
			# vertical component
			aux_v +=  (v[i+1,j] - v[i,j]  )*dy_dx # e
			aux_v += -(v[i,j]   - v[i-1,j])*dy_dx # w
			aux_v +=  (v[i,j+1] - v[i,j]  )*dy_dx # n
			aux_v += -(v[i,j]   - v[i,j-1])*dy_dx # w
	
			# diffusion v component
			Dv[i,j] = aux_v/volume

	return Du, Dv

def calcR(u,v,nu,mesh):
	'''
	calcR(Cu,Cv,Du,Dv,nu): Compute the R term in NS equation
	
	This function return the R term for each NS equation in the 2D
	problem as  R(n) = -C+D*nu
	
	Input:
	  Cu: double 2D array, convection term in the x equation
	  Cv: double 2D array, convection term in the y equation
	  Du: double 2D array, diffusion term in the x equation
	  Dv: double 2D array, diffusion term in the y equation
	
	Output: 
	  Ru,Rv: R term in x and y equations respectively
	
	Example of use
	  [Ru,Rv]=calcR(Cu,Cv,Du,Dv)
	'''
	# calculate convection and diffusion terms
	Cu, Cv = convection_new(mesh,u,v)
	Du, Dv = diffusion(mesh,u,v)

	# R = -C(u)u + nu*D(u)
	Ru = -Cu + nu*Du
	Rv = -Cv + nu*Dv

	return Ru, Rv

def solve_poisson(up,vp,mesh):
	'''
	'''
	def _poisson(d,mesh):
		'''
		poisson(b,N): Solve the poisson equation
		
		Input:
		  b: algebraic vector (N*N) of the divergence matrix
		  N: number of nodes of the mesh
		Output:
		  p: vector (N*N) with the results of the poisson equation
		  A: matrix (N^2,N^2) with the values of the pressure nodes
		Example of use
			  diverg(b,64)
		'''
		N = mesh['N'][0]
		M = mesh['N'][1]

		# Reshape d to N*M as b vector
		b = np.zeros((N*M,),np.double)
		for i in range(1,N+1):
			for j in range(1,M+1): # Loops all the mesh
				idx    = (j-1)*N + (i-1)
				b[idx] = d[i,j]
#		print('Residual: ',abs(np.sum(b)))
		assert abs(np.sum(b)) < 1e-10

		# System matrix
		A = np.zeros((N*M,N*M),np.double)
		np.fill_diagonal(A,-4.) # as in diapo 45

		for i in range(N):
			for j in range(M): # Loops all the mesh
				idx = i*M + j
				if j>0 and j < (M-1):
					A[idx+1,idx]   = 1.
					A[idx-1,idx]   = 1.
				if i>0 and i < (N-1):
					A[idx+N,idx]   = 1.
					A[idx-N,idx]   = 1.
				if i == 0:   # Bottom
					A[N+j,idx]     = 1.
					A[N*M-N+j,idx] = 1.
				if j == 0:   # Left
					A[idx+M-1,idx] = 1.
					A[idx+1  ,idx] = 1.
				if i == N-1: # Top
					A[idx-N,idx]   = 1.
					A[j    ,idx]   = 1.
				if j == M-1: # Right
					A[idx-M+1,idx] = 1.
					A[idx-1,  idx] = 1.
		A[0,0] = -5.

		# Solve the system x = A^-1*b
		x = np.matmul(np.linalg.inv(A),b)

		# Obtain output vector
		o = np.zeros((N+2,M+2),np.double)
		for i in range(1,N+1):
			for j in range(1,M+1): # Loops all the mesh
				idx    = (j-1)*N + (i-1)
				o[i,j] = x[idx]

		# Return the solution as an array
		return o

	# Compute div(up)
	d = divergence(up,vp,mesh)

	# Solve the poisson equation
	ps = _poisson(d,mesh) # lap(ps) = div(up)

	return halo_update(ps)


## EIGENCD ROUTINES

def calcG(x,a,b,c,x0,x1,f0,f1):
	'''Trias, F. X. and Lehmkuhl, O., “A self-adaptive strategy for the
	 time integration of Navier-Stokes equations,” Aug 2011.'''
	Q = (x-x0)*(x-x1)
	L = f0+(x-x0)*(f1-f0)/(x1-x0)
	return (a*x**2+b*x+c)*Q+L

def calcTopt(phi,phil,c,tl):
	'''Trias, F. X. and Lehmkuhl, O., “A self-adaptive strategy for the
	 time integration of Navier-Stokes equations,” Aug 2011.'''
	Topt = None
	if phi >= 0 and phi < phil[0]:
		Topt = calcG(phi,0,c[0],c[1],0,phil[0],4/3,tl)
	if phi >= phil[0] and phi <= np.pi/2:
		Topt = calcG(phi,c[2],c[3],c[4],phil[0],0.5*np.pi,tl,1)
	return Topt

def calcKopt(phi,phil,c,k):
	''''Trias, F. X. and Lehmkuhl, O., “A self-adaptive strategy for the
	 time integration of Navier-Stokes equations,” Aug 2011.'''
	Kopt = 1
	if phil[0]<phi and phi <= phil[1]:
	 	Kopt = calcG(phi,c[5],c[6],c[7],phil[0],phil[1],1,k[0])
	if phil[1]<phi and phi <= phil[2]:
		Kopt = calcG(phi,c[8],c[9],c[10],phil[1],phil[2],k[0],k[1])
	if phil[2]<phi and phi <= np.pi/2:
		Kopt = calcG(phi,c[11],c[12],c[13],phil[2],np.pi/2,k[1],0)
	return Kopt

def boundCD(u,v,nu,mesh):
	''''Trias, F. X. and Lehmkuhl, O., “A self-adaptive strategy for the
	 time integration of Navier-Stokes equations,” Aug 2011.'''
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	dx = mesh['delta'][0]
	dy = mesh['delta'][1]
	cx = mesh['cx']
	cy = mesh['cy']
	sx = mesh['sx']
	sy = mesh['sy']

	a,b = -1e11, -1e11

	for i in range(1,N+1):
		for j in range(1,M+1): # Loops all the mesh
			# Bounds for u
			dn = dx/(cy[j+1] - cy[j]  )
			ds = dx/(cy[j]   - cy[j-1])
			de = dy/(sx[i+1] - sx[i]  )
			dw = dy/(sx[i]   - sx[i-1])
			cn = 0.5*abs(u[i,j+1] + u[i,j]  )*dx
			cs = 0.5*abs(u[i,j]   + u[i,j-1])*dx
			ce = 0.5*abs(u[i+1,j] + u[i,j]  )*dy
			cw = 0.5*abs(u[i,j]   + u[i-1,j])*dy

			a  = max(a,1/(dx*dy)*nu*(dn+ds+de+dw))
			b  = max(b,(0.5/(dx*dy))*(cn+cs+ce+cw))

			# Bounds per v
			dn = dx/(sy[j+1] - sy[j]  )
			ds = dx/(sy[j]   - sy[j-1])
			de = dy/(cx[i+1] - cx[i]  )
			dw = dy/(cx[i]   - cx[i-1])
			cn = 0.5*abs(v[i,j+1] + v[i,j]  )*dx
			cs = 0.5*abs(v[i,j]   + v[i,j-1])*dx
			ce = 0.5*abs(v[i+1,j] + v[i,j]  )*dy
			cw = 0.5*abs(v[i,j]   + v[i-1,j])*dy

			a  = max(a,1/(dx*dy)*nu*(dn+ds+de+dw))
			b  = max(b,(0.5/(dx*dy))*(cn+cs+ce+cw))

	return a,b 

def eigenCD(f,nu,u,v,gamma,mesh):
	''''Trias, F. X. and Lehmkuhl, O., “A self-adaptive strategy for the
	 time integration of Navier-Stokes equations,” Aug 2011.'''
	c = np.array([0.0647998,-0.386022,3.72945,-9.38143,7.06574,2403400,
		           -5018490,  2620140,   2945,-6665.76,3790.54,4.80513,
		           -16.9473,  15.0155],np.double) # Trias and Lehmkul
	phil = np.array([np.arctan(164./99.),np.pi/3.,(3./5.)**2*np.pi],np.double)
	k    = np.array([0.73782212, 0.44660387],np.double)
	tl   = 0.9302468;

	a, b = boundCD(u,v,nu,mesh)

	phi  = np.arctan(b/a)
	Topt = calcTopt(phi,phil,c,tl)

	dt   = Topt/np.sqrt(a**2. + b**2.)
	beta = calcKopt(phi,phil,c,k)

	return f*dt, beta


# RK ROUTINES
ERK_tableau = {
    'midpoint_RK2' : { # Alya
        'diagonal' : True,
        'order':    2,
        's' : 2,
        'a' : np.array([[0.,0.],[0.5,0.]],np.double),
        'b' : np.array([0.,1.],np.double),
        'c' : np.array([0.,0.5],np.double),
        'ROS':np.array([0.0000000789671728,0.0000000314319595,-0.0000017790081110,-0.0000222342968956,0.0003144615858365,0.0037647948617420,-0.0352139592445843,-0.2535832738741128,1.1859861427673575,2.8468539726219513],np.double)
    },
    'kutta_RK3' : {
        'diagonal' : False,
        'order':    3,
        's' : 3,
        'a' : np.array([[0.,0.,0.],[1./2.,0.,0.],[-1.,2.,0.]],np.double),
        'b' : np.array([1./6.,2./3.,1./6.],np.double),
        'c' : np.sum(np.array([[0.,0.,0.],[1./2.,0.,0.],[-1.,2.,0.]],np.double),axis=1),
        'ROS':np.array([0.0025036520382173,-0.0106785369531126,0.0045230875006834,0.0224572748768649,-0.0008368111281387,-0.0115104208048807,0.0854336568144740,-0.5296671138386586,0.2144836576745466,3.6633661734060912],np.double)
    },
    'heun_RK3' : {
        'diagonal' : True,
        'order':    3,
        's' : 3,
        'a' : np.array([[0.,0.,0.],[1./3.,0.,0.],[0.,2./3.,0.]],np.double),
        'b' : np.array([1./4.,0,3./4.],np.double),
        'c' : np.sum(np.array([[0.,0.,0.],[1./3.,0.,0.],[0.,2./3.,0.]],np.double),axis=1),
        'ROS':np.array([0.0025036520382173,-0.0106785369531126,0.0045230875006834,0.0224572748768649,-0.0008368111281387,-0.0115104208048807,0.0854336568144740,-0.5296671138386586,0.2144836576745466,3.6633661734060912],np.double)
    },
    'ralston_RK3' : {
        'diagonal' : True,
        'order':    3,
        's' : 3,
        'a' : np.array([[0.,0.,0.],[0.5,0.,0.],[0.,3./4.,0.]],np.double),
        'b' : np.array([1./6.,1./6.,2./3.],np.double),
        'c' : np.sum(np.array([[0.,0.,0.],[0.5,0.,0.],[0.,3./4.,0.]],np.double),axis=1),
        'ROS':np.array([0.0025036520382173,-0.0106785369531126,0.0045230875006834,0.0224572748768649,-0.0008368111281387,-0.0115104208048807,0.0854336568144740,-0.5296671138386586,0.2144836576745466,3.6633661734060912],np.double)

    },
    'SSP_RK3' : { # Alya
        'diagonal' : False,
        'order':    3,
        's' : 3,
        'a' : np.array([[0.,0.,0.],[1.,0.,0.],[1./4.,1./4.,0.]],np.double),
        'b' : np.array([1./6.,1./6.,2./3.],np.double),
        'c' : np.array([0.,1.,0.5],np.double),
        'ROS':np.array([0.0025036520382173,-0.0106785369531126,0.0045230875006834,0.0224572748768649,-0.0008368111281387,-0.0115104208048807,0.0854336568144740,-0.5296671138386586,0.2144836576745466,3.6633661734060912],np.double)
    },
    'classic_RK4' : { # Alya
        'diagonal' : True,
        'order':    4,
        's' : 4,
        'a' : np.array([[0.,0.,0.,0.],[0.5,0.,0.,0.],[0.,0.5,0.,0.],[0.,0.,1.,0.]],np.double),
        'b' : np.array([1./6.,1./3.,1./3.,1./6.],np.double),
        'c' : np.array([0.,0.5,0.5,1.],np.double),
        'ROS':np.array([0.0029780007240036,-0.0072511833405441,-0.0158126137147068,0.0651367605636128,0.0097650938421710,-0.3113960849920550,0.1658919977817575,0.6596882558665331,-0.2205111894300658,3.3385664853141104],np.double)
    },
    'SSP_RK4' : { 
        'diagonal' : False,
        'order':    4,
        's' : 10,
        'a' : np.transpose(np.array([[0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,0.0,1.0/6.0 ,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,0.0,0.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],np.double)),
        'b' : np.array([1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0],np.double),
        'c' : np.sum(np.array([[0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,0.0,1.0/6.0 ,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,0.0,0.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],np.double),axis=1),
        'ROS':np.array([-0.0001397596506854,0.1080968836274558,0.0607387998441175,-0.8193990507954670,-0.3566660433677002,2.2687939369013379,0.5608979904302257,-2.9579645951719056,2.6190695049784596,12.7152941245561681],np.double)
    },
    '3p5q' : {
        'diagonal' : False,
        'order':    3,
        's' : 4,
        'a' : np.array([[0.,0.,0.,0.],[0.375,0.,0.,0.],[0.91666667,-0.66666667,0.,0.],[-0.08333333,1.83333333,-0.75,0.]],np.double),
        'b' : np.array([0.111,0.889,-0.222,0.222],np.double),
        'c' : np.array([0,0.375,0.25,1],np.double),
        'ROS':np.array([0.0029864120078651,-0.0072385310205102,-0.0159351810123712,0.0650890801056286,0.0103262263302469,-0.3113290760984786,0.1646383268587967,0.6600072405820170,-0.2200851929937220,3.3392663309774933],np.double),
    },
    '4p7q' : {
        'diagonal' : False,
        'order':    4,
        's' : 6,
        'a' : np.array([[ 0., 0., 0.,0.,0.,0.],[0.23593377,0.,0.,0.,0.,0.],[0.34750736,-0.13561935,0.,0.,0.,0.],
                        [-0.20592852,1.89179077,-0.89775024,0.,0.,0.],[-0.09435493,1.75617141,-0.9670785,0.06932826,0.,0.],
                        [ 0.14157883,-1.17039696,1.30579112,-2.20354137,2.92656838,0.]],np.double),
        'b' : np.array([0.07078942,0.87808571,-0.44887512,-0.44887512,0.87808571,0.07078942],np.double),
        'c' : np.sum(np.array([[ 0., 0., 0.,0.,0.],[0.23593377,0.,0.,0.,0.],[ 0.34750736,-0.13561935,0.,0.,0.],
                        [-0.20592852,1.89179077,-0.89775024,0.,0.],[-0.09435493,1.75617141,-0.9670785,0.06932826,0.],
                        [ 0.14157883,-1.17039696,1.30579112,-2.20354137,2.92656838]],np.double),axis=1)
    },
}

LSERK_tableau = {
    'RK3LR' : {
        # Williamson 180.  (least round-off error scheme)
        # Low-Storage Runge-Kutta Schemes. 
        # Journal of Computational Physics 35, 48–56.
        'order':    3,
        's' : 3,
        'A' : np.array([0.0,-6.573094670655647,0.046582819666687],np.double),
        'B' : np.array([1.13541,0.15232,0.96366],np.double),
        'c' : np.array([0.0,1.13541,0.15232],np.double),
        'ROS':np.array([0.0010524405834221,-0.0019595009557986,-0.0069816801888216,0.0025556661622857,0.0299173582412022,0.0371044658054973,0.0727589057677445,-0.5065285883240086,-0.3459193717329608,3.5117282302810295],np.double),
    },
    'RK3LT' : {
        # Williamson 180.  (least truncation error scheme)
        # Low-Storage Runge-Kutta Schemes. 
        # Journal of Computational Physics 35, 48–56.
        'order':    3,
        's' : 3,
        'A' : np.array([0.0,-0.637676219230114,-1.306629764111419],np.double),
        'B' : np.array([0.45736,0.9253,0.39383],np.double),
        'c' : np.array([0.0,0.45736,0.9253],np.double),
        'ROS':np.array([0.0112196303442080,-0.0482935928361276,0.0244383515187455,0.1173092951784913,-0.0869378227697190,-0.1067043228924655,0.1690765223813385,-0.5343068173402841,0.4627524998887296,3.6497802415748857],np.double),
    },
    'RK4s5' : {
        # Carpenter, M.H., Kennedy, C.A., 1994. 
        # Fourth-Order 2N-Storage Runge- Kutta Schemes (Technical Memorandum (TM) 
        # No. NASA-TM-109112). NASA.
        'order':    4,
        's' : 5,
        'A' : np.array([0.0,-0.4178904745,-1.192151694643,-1.697784692471,-1.514183444257],np.double),
        'B' : np.array([0.1496590219993,0.3792103129999,0.8229550293869,0.6994504559488,0.1530572479681],np.double),
        'c' : np.array([0.0,0.1496590219993,0.3704009573644,0.6222557631345,0.9582821306748],np.double),
        'ROS':np.array([0.0041864709926662,-0.0041951887629049,-0.0493572588196629,0.0432966265347427,0.2652127003214184,-0.2048684891867446,-0.7353149529110932,0.4242702297487767,1.1990824211946132,4.6475934571756623],np.double),
    },
    'RK4s6' : {
        # Allampalli, V., Hixon, R., Nallasamy, M., Sawyer, S.D., 2009. 
        # High-accuracy large-step explicit Runge–Kutta (HALE-RK) schemes for 
        # computational aeroacoustics. Journal of Computational Physics 228, 3837–3850. 
        'order':    4,
        's' : 6,
        'A' : np.array([0.000000000000,-0.691750960670,-1.727127405211,-0.694890150986,-1.039942756197,-1.531977447611],np.double),
        'B' : np.array([0.122000000000,0.477263056358,0.381941220320,0.447757195744,0.498614246822,0.186648570846],np.double),
        'c' : np.array([0.000000000000,0.122000000000,0.269115878630,0.447717183551,0.749979795490,0.898555413085],np.double),
        'ROS':np.array([-0.0046838947710738,-0.0102821002313504,0.0895822851676951,0.0157576435727931,-0.4313332285687259,0.0781500406857502,0.8968764564546161,-0.0240276188089227,-1.0980542012827061,4.6152231791195808],np.double),
    },
    'RK4s7' : {
        # Allampalli, V., Hixon, R., Nallasamy, M., Sawyer, S.D., 2009. 
        # High-accuracy large-step explicit Runge–Kutta (HALE-RK) schemes for 
        # computational aeroacoustics. Journal of Computational Physics 228, 3837–3850. 
        'order':    4,
        's' : 7,
        'A' : np.array([0.0,-0.647900745934,-2.704760863204,-0.460080550118,-0.500581787785,-1.906532255913,-1.450000000000],np.double),
        'B' : np.array([0.117322146869,0.503270262127,0.233663281658,0.283419634625,0.540367414023,0.371499414620,0.136670099385],np.double),
        'c' : np.array([0.000000000000,0.117322146869,0.294523230758,0.305658622131,0.582864148403,0.858664273599,0.868664273599],np.double),
        'ROS':np.array([0.0070347459893183,-0.0138792760918709,-0.0693390395129110,0.1515553796771410,0.2655131951960983,-0.6438791807806871,-0.4138205848733530,1.1170229378390459,-0.1930750120907563,4.9480337876746532],np.double),
    },
    'RK4s12' : {
        # Niegemann, J., Diehl, R., Busch, K., 2012. 
        # Efficient low-storage Runge–Kutta schemes with optimized stability regions. 
        # Journal of Computational Physics 231, 364–372. 
        'order':    4,
        's' : 12,
        'A' : np.array([0.0000000000000000,-0.0923311242368072,-0.9441056581158819,-4.3271273247576394,-2.1557771329026072,-0.9770727190189062,-0.7581835342571139,-1.7977525470825499,-2.6915667972700770,-4.6466798960268143,-0.1539613783825189,-0.5943293901830616],np.double),
        'B' : np.array([0.0650008435125904,0.0161459902249842,0.5758627178358159,0.1649758848361671,0.3934619494248182,0.0443509641602719,0.2074504268408778,0.6914247433015102,0.3766646883450449,0.0757190350155483,0.2027862031054088,0.2167029365631842],np.double),
        'c' : np.array([0.0000000000000000 ,0.0650008435125904 ,0.0796560563081853 ,0.1620416710085376 ,0.2248877362907778 ,0.2952293985641261 ,0.3318332506149405 ,0.4094724050198658 ,0.6356954475753369 ,0.6806551557645497 ,0.7143773712418350 ,0.9032588871651854],np.double),
        'ROS':np.array([0.0002905025782957,0.0764948195374607,-0.2963472324787501,-0.1111489019402667,1.3525911892197995,-0.5000412061660995,-1.8035946630990043,1.5774468512239213,-1.7996356094456889,7.0402579498437277],np.double),
    },
    'RK4s13' : {
        # Niegemann, J., Diehl, R., Busch, K., 2012. 
        # Efficient low-storage Runge–Kutta schemes with optimized stability regions. 
        # Journal of Computational Physics 231, 364–372. 
        'order':    4,
        's' : 13,
        'A' : np.array([0.0000000000000000,-0.6160178650170565,-0.4449487060774118,-1.0952033345276178,-1.2256030785959187,-0.2740182222332805,-0.0411952089052647,-0.1797084899153560,-1.1771530652064288,-0.4078831463120878,-0.8295636426191777,-4.7895970584252288,-0.6606671432964504],np.double),
        'B' : np.array([0.0271990297818803,0.1772488819905108,0.0378528418949694,0.6086431830142991,0.2154313974316100,0.2066152563885843,0.0415864076069797,0.0219891884310925,0.9893081222650993,0.0063199019859826,0.3749640721105318,1.6080235151003195,0.0961209123818189],np.double),
        'c' : np.array([0.0000000000000000,0.0271990297818803,0.0952594339119365,0.1266450286591127,0.1825883045699772,0.3737511439063931,0.5301279418422206,0.5704177433952291,0.5885784947099155,0.6160769826246714,0.6223252334314046,0.6897593128753419,0.9126827615920843],np.double),
        'ROS':np.array([0.0478181843634897,-0.1049881015230035,-0.2813854033879319,0.6129482363910974,0.5117552100935840,-1.1206489660776351,0.1761508368926473,-0.3267746875399357,-0.0488576866394565,13.3479930476407560],np.double),
    },
    'RK4s14' : {
        # Niegemann, J., Diehl, R., Busch, K., 2012. 
        # Efficient low-storage Runge–Kutta schemes with optimized stability regions. 
        # Journal of Computational Physics 231, 364–372. 
        'order':    4,
        's' : 14,
        'A' : np.array([0.0000000000000000,-0.7188012108672410,-0.7785331173421570,-0.0053282796654044,-0.8552979934029281,-3.9564138245774565,-1.5780575380587385,-2.0837094552574054,-0.7483334182761610,-0.7032861106563359,0.0013917096117681,-0.0932075369637460,-0.9514200470875948,-7.1151571693922548],np.double),
        'B' : np.array([0.0367762454319673,0.3136296607553959,0.1531848691869027,0.0030097086818182,0.3326293790646110,0.2440251405350864,0.3718879239592277,0.6204126221582444,0.1524043173028741,0.0760894927419266,0.0077604214040978,0.0024647284755382,0.0780348340049386,5.5059777270269628],np.double),
        'c' : np.array([0.0000000000000000 ,0.0367762454319673 ,0.1249685262725025 ,0.2446177702277698 ,0.2476149531070420 ,0.2969311120382472 ,0.3978149645802642 ,0.5270854589440328 ,0.6981269994175695 ,0.8190890835352128 ,0.8527059887098624 ,0.8604711817462826 ,0.8627060376969976 ,0.8734213127600976],np.double),
        'ROS': np.array([0.005249823415646,-0.017257898556630,-0.16004030798344308,0.096889705951851,-0.071662423295669,-0.142598183112801,-0.096761269488307,-0.672886946891750,4.256069377037732,16.979060335149779],np.double),
    },
    'LDDRK2s5' : {
        # Stanescu, D., Habashi, W.G., 1998. 
        # 2N-Storage Low Dissipation and Dispersion Runge-Kutta Schemes for Computational Acoustics. 
        # Journal of Computational Physics 143, 674–681.
        'order':    4,
        's' : 5,
        'A' : np.array([0.0,-0.6913065,-2.655155,-0.8147688,-0.6686587],np.double),
        'B' : np.array([0.1,0.75,0.7,0.479313,0.310392],np.double),
        'c' : np.array([0.0,0.1,0.3315201,0.4577796,0.8666528],np.double),
        'ROS':np.array([-0.0033193246944026,0.0090245476315176,0.0135404659708355,-0.0752345654639393,0.0890375506745488,0.1848475550403259,-0.5402980203740597,-0.1216142024236540,0.6304428063064985,4.0542006145148806],np.double),
    },
    'LDDRK4s6' : {
        # Stanescu, D., Habashi, W.G., 1998. 
        # 2N-Storage Low Dissipation and Dispersion Runge-Kutta Schemes for Computational Acoustics. 
        # Journal of Computational Physics 143, 674–681.
        'order':    4,
        's' : 6,
        'A' : np.array([0.0,-0.4919575,-0.8946264,-1.5526678,-3.4077973,-1.0742640],np.double),
        'B' : np.array([0.1453095,0.4653797,0.4675397,0.7795279,0.3574327,0.15],np.double),
        'c' : np.array([0.0,0.1453095,0.3817422,0.6367813,0.7560744,0.9271047],np.double),
        'ROS':np.array([-0.0020060398871356,0.0036436308944101,0.0411655482875007,-0.0445542007108853,-0.2910653555595412,0.2476499071911762,0.7414158680216698,-0.4560744514190780,-0.5013444373493209,4.5127851780227139],np.double),
    },


}

def convert(t):
	'''convert(t): Convert the LSERK tableau to standard RK tableau
	
	Niegemann, J., Diehl, R., and Busch, K., “Efficient low-storage runge–kutta schemes with
	optimized stability regions,” Journal of computational physics, vol. 231, no. 2, p. 364–372,
	2012.'''
	# Initialize variables
	a = np.zeros((t['s'],t['s']),np.double)
	b = np.zeros((t['s'],),np.double)
	c = np.zeros((t['s'],),np.double)
	# Compute b
	b[t['s']-1] = t['B'][t['s']-1]
	for i in reversed(range(1,t['s'])):
		b[i-1] = t['A'][i]*b[i] + t['B'][i-1]
	# Compute A
	for i in reversed(range(t['s'])):
		c[i] = t['c'][i]
		for j in reversed(range(t['s']-1)):
			if j >= i:
				a[i,j] = 0.0
			elif i == j + 1:
				a[i,j] = t['B'][j]
			else:
				a[i,j] = t['A'][j+1]*a[i,j+1] + t['B'][j]

	print(a)
	print(b)
	print(c)
	return {'s':t['s'],'a':a,'b':b,'c':c}

def compute_region_of_stability(t):
	'''compute_region_of_stability(t): Compute stability function of a RK scheme
	
	Niegemann, J., Diehl, R., and Busch, K., “Efficient low-storage runge–kutta schemes with
	optimized stability regions,” Journal of computational physics, vol. 231, no. 2, p. 364–372,
	2012.'''
	s = t['s']
	# Compute the gamma coefficients
	# to be evaluated with polyval
	gamma = np.zeros((s+1,),np.double)
	# Last term is always 1
	gamma[0] = 1.0
	# Compute the remaining terms
	for i in range(1,s+1):
		aux = t['b'].copy()
		for j in range(i-1):
			aux = np.matmul(aux,t['a'])
		gamma[i] = np.dot(aux,t['c'])
	return np.flip(gamma)
	

def plot_region_of_stability(t):
	'''plot_region_of_stability(t): plot the ROS of a ERK method
	'''
	gamma = compute_region_of_stability(t)
	x = np.linspace(-6,6,100)
	y = np.linspace(-6,6,100)
	xx,yy = np.meshgrid(x,y)
	z = np.abs(np.polyval(gamma,xx + yy*1j))
	plt.figure()
	plt.contour(x,y,z,[1.0])
	plt.plot(x,np.zeros_like(x),'--k')
	plt.plot(np.zeros_like(y),y,'--k')
	plt.xlim(-6,1.0)
	plt.ylim(-6,6)

def calcR_ERK_consistent(u,v,dt,nu,t,mesh):
	'''
	Computation of the R term using Runge-Kutta method.
	
	Implementing the consistent formulation, i.e., solving
	the pressure projection at each substep according to the
	set of equations 4 of Cappuano et al.

	Ref:
		Capuano, F., Coppola, G., Chiatto, M., de Luca, L., 2016. 
		Approximate Projection Method for the Incompressible Navier–Stokes Equations. 
		AIAA Journal 54, 2179–2182. https://doi.org/10.2514/1.J054569
	'''
	# Initialization of variables
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	dx = mesh['delta'][0]
	dy = mesh['delta'][1]
	
	Ru, Rv   = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	Rus, Rvs = np.zeros((N+2,M+2,t['s']),np.double), np.zeros((N+2,M+2,t['s']),np.double)

	# Loop the stages
	for ist in range(t['s']):
		Russ, Rvss = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
		for jst in range(ist):
			Russ += t['a'][ist,jst]*Rus[:,:,jst]
			Rvss += t['a'][ist,jst]*Rvs[:,:,jst]

		# Compute up of the stage
		usp = halo_update(u + dt*Russ)
		vsp = halo_update(v + dt*Rvss)

		if ist == 0:
			us = usp
			vs = vsp
		else:
			# Solve the poisson equation to obtain the pseudo-pressure
			pss = solve_poisson(usp,vsp,mesh)

			# Obtain the gradient of the pseudo-pressure
			gpssx, gpssy = gradient(pss,mesh)

			# Compute us
			# ci is included inside ps as eq. 4 of Cappuano et al.
			us = usp - gpssx # u^(n+1) = up - grad(ps)
			vs = vsp - gpssy # v^(n+1) = vp - grad(ps)		

		# Compute R
		Rus[:,:,ist], Rvs[:,:,ist] = calcR(us,vs,nu,mesh)

		# Accumulate R
		Ru += t['b'][ist]*Rus[:,:,ist]
		Rv += t['b'][ist]*Rvs[:,:,ist]

	return Ru, Rv

def calcR_ERK_diagonal_consistent(u,v,dt,nu,t,mesh):
    '''
    Computation of the R term using Runge-Kutta method.
    
    Implementing the consistent formulation, i.e., solving
    the pressure projection at each substep according to the
    set of equations 4 of Cappuano et al.

    Ref:
        Capuano, F., Coppola, G., Chiatto, M., de Luca, L., 2016. 
        Approximate Projection Method for the Incompressible Navier–Stokes Equations. 
        AIAA Journal 54, 2179–2182. https://doi.org/10.2514/1.J054569
    '''
    # Initialization of variables
    N  = mesh['N'][0]
    M  = mesh['N'][1]
    dx = mesh['delta'][0]
    dy = mesh['delta'][1]
    
    Ru, Rv   = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
    Rus, Rvs = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)

    # Loop the stages
    for ist in range(t['s']):

        if ist>0:
            Rus = t['a'][ist,ist-1]*Rus
            Rvs = t['a'][ist,ist-1]*Rvs

        # Compute up of the stage
        usp = halo_update(u + dt*Rus)
        vsp = halo_update(v + dt*Rvs)

        if ist == 0:
            us = usp
            vs = vsp
        else:
            # Solve the poisson equation to obtain the pseudo-pressure
            pss = solve_poisson(usp,vsp,mesh)

            # Obtain the gradient of the pseudo-pressure
            gpssx, gpssy = gradient(pss,mesh)

            # Compute us
            # ci is included inside ps as eq. 4 of Cappuano et al.
            us = usp - gpssx # u^(n+1) = up - grad(ps)
            vs = vsp - gpssy # v^(n+1) = vp - grad(ps)        

        # Compute R
        Rus, Rvs = calcR(us,vs,nu,mesh)

        # Accumulate R
        Ru += t['b'][ist]*Rus
        Rv += t['b'][ist]*Rvs

    return Ru, Rv


def calcR_ERK_aproximate(u,v,ps,psm1,psm2,approximation,dt,nu,t,mesh):
	'''
	Computation of the R term using a general Runge-Kutta method.
	
	Implementing the approximate formulation, i.e., implementing the set
	Karam et a. and using the approximation for
	the projection of the pseudo-pressure proposed in the paper.

	Ref:
		Karam, M., Sutherland, J. C., and Saad, T., “Low-cost runge-kutta integrators for in-
		compressible flow simulations,” Journal of computational physics, vol. 443, no. 110518,
		p. 110518, 2021.
	'''
	# Initialization of variables
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	dx = mesh['delta'][0]
	dy = mesh['delta'][1]
	
	Ru, Rv   = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	Rus, Rvs = np.zeros((N+2,M+2,t['s']),np.double), np.zeros((N+2,M+2,t['s']),np.double)
	phih      = np.zeros((N+2,M+2,t['s']-1),np.double)

	if approximation.lower() in ['approximated_2']:
		if t['s']==2:
			phih[:,:,0] = halo_update(ps)
		elif t['s']==3:
			phih[:,:,0] = halo_update(0.5*(3*ps-psm1))
			phih[:,:,1] = halo_update(0.5*(3*ps-psm1) + t['a'][1,0]*(ps-psm1))
		elif t['s']==4:
			phih[:,:,0] = halo_update(1./6.*(11*ps-7*psm1+2*psm2))
			phih[:,:,1] = halo_update(1./6.*(11*ps-7*psm1+2*psm2) + t['a'][1,0]*(2*ps-3*psm1+psm2))
			phih[:,:,2] = halo_update(1./6.*(11*ps-7*psm1+2*psm2) + (t['a'][2][0]+t['a'][2][1])*(2*ps-3*psm1+psm2) + (t['a'][2][1]*t['a'][1][0])*(ps-2*psm1+psm2)) 
		else:
			print('t[s]=',t['s'])
			exit('Approximated_2 not existing yet for more than 4 stages')
	else: 
		for st in range(t['s']-1):
			phih[:,:,st] = halo_update(ps)

	

	# Loop the stages
	for ist in range(t['s']):
		Russ, Rvss = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
		#Rutest, Rvtest = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
		for jst in range(0,ist):
			gpssx, gpssy = gradient(phih[:,:,jst],mesh)
			Russ += t['a'][ist,jst]*(Rus[:,:,jst] - gpssx/dt)
			Rvss += t['a'][ist,jst]*(Rvs[:,:,jst] - gpssy/dt)
		
		# Compute up of the stage
		us = halo_update(u + dt*Russ)
		vs = halo_update(v + dt*Rvss)
	
		div = np.max(divergence(us,vs,mesh))
		#print("Div stg=",ist,"=",div)
		
		# Compute R
		Rus[:,:,ist], Rvs[:,:,ist] = calcR(us,vs,nu,mesh)

		# Accumulate R
		Ru += t['b'][ist]*Rus[:,:,ist]
		Rv += t['b'][ist]*Rvs[:,:,ist]

	return Ru, Rv

def calcR_ERK_diagonal_aproximate(u,v,ps,psm1,psm2,approximation,dt,nu,t,mesh):
    '''
	Computation of the R term using a diagonal Runge-Kutta method.
	
	Implementing the approximate formulation, i.e., implementing the set
	Karam et a. and using the approximation for
	the projection of the pseudo-pressure proposed in the paper.

	Ref:
		Karam, M., Sutherland, J. C., and Saad, T., “Low-cost runge-kutta integrators for in-
		compressible flow simulations,” Journal of computational physics, vol. 443, no. 110518,
		p. 110518, 2021.
	'''
    # Initialization of variables
    N  = mesh['N'][0]
    M  = mesh['N'][1]
    dx = mesh['delta'][0]
    dy = mesh['delta'][1]
    
    Ru, Rv   = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
    Rus, Rvs = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
    phih      = np.zeros((N+2,M+2,t['s']-1),np.double)

    if approximation.lower() in ['approximated_2']:
        if t['s']==2:
            phih[:,:,0] = halo_update(ps)
        if t['s']==3:
            phih[:,:,0] = halo_update(0.5*(3*ps-psm1))
            phih[:,:,1] = halo_update(0.5*(3*ps-psm1) + t['a'][1,0]*(ps-psm1))
        if t['s']==4:
            phih[:,:,0] = halo_update(1./6.*(11*ps-7*psm1+2*psm2))
            phih[:,:,1] = halo_update(1./6.*(11*ps-7*psm1+2*psm2) + t['a'][1,0]*(2*ps-3*psm1+psm2))
            phih[:,:,2] = halo_update(1./6.*(11*ps-7*psm1+2*psm2) + (t['a'][2][0]+t['a'][2][1])*(2*ps-3*psm1+psm2) + (t['a'][2][1]*t['a'][1][0])*(ps-2*psm1+psm2)) 
        else:
            exit('Approximated_2 not existing yet for more than 4 stages')
    else: 
        for st in range(t['s']-1):
            phih[:,:,st] = halo_update(ps)

    # Loop the stages
    for ist in range(t['s']):
        
        if ist>0:
            gpssx, gpssy = gradient(phih[:,:,ist-1],mesh)
            Rus = t['a'][ist,ist-1]*(Rus - gpssx/dt)
            Rvs = t['a'][ist,ist-1]*(Rvs - gpssy/dt)
        
        # Compute up of the stage
        us = halo_update(u + dt*Rus)
        vs = halo_update(v + dt*Rvs)
            
        # Compute R
        Rus, Rvs = calcR(us,vs,nu,mesh)

        # Accumulate R
        Ru += t['b'][ist]*Rus
        Rv += t['b'][ist]*Rvs

    return Ru, Rv


def calc_LSERK_consistent(u,v,dt,nu,t,mesh):
	'''
	Computation of the R term using low storage Runge-Kutta method.
	
	Implementing the consistent formulation, i.e., solving
	the pressure projection at each substep according to the
	set of equations 4 of Cappuano et al.

	Ref:
		Kennedy, C. A., Carpenter, M. H., and Lewis, R., “Low-storage, explicit runge–kutta
		schemes for the compressible navier–stokes equations,” Applied Numerical Mathematics,
		vol. 35, no. 3, pp. 177–219, 2000.
	'''
	# Initialization of variables
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	
	Ku1, Ku2 = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	Kv1, Kv2 = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	usp, vsp = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)

	Ku1, Kv1 = u.copy(), v.copy()
	# Loop the stages
	for ist in range(t['s']):
		# Evaluate R
		Rus, Rvs = calcR(Ku1,Kv1,nu,mesh)
		
		# Compute up of the stage
		usp = halo_update(t['A'][ist]*Ku2 + dt*Rus)
		vsp = halo_update(t['A'][ist]*Kv2 + dt*Rvs)

		# Solve the poisson equation to obtain the pseudo-pressure
		pss = solve_poisson(usp,vsp,mesh)

		# Obtain the gradient of the pseudo-pressure
		gpssx, gpssy = gradient(pss,mesh)

		# Compute K2
		Ku2 = usp - gpssx
		Kv2 = vsp - gpssy

		# Update K1
		Ku1 = Ku1 + t['B'][ist]*Ku2
		Kv1 = Kv1 + t['B'][ist]*Kv2

	return Ku1, Kv1, pss

def calc_LSERK_approximate(u,v,ps,psm1,approximation,dt,nu,t,mesh):
	'''
	Computation of the R term using low storage Runge-Kutta method.
	
	THIS FUNCTION DOES NOT WORK. Implementing the approximate formulation, i.e., solving
	the pressure projection at each substep according to the
	set of equations of Karam et al.

	Ref:
		Karam, M., Sutherland, J. C., and Saad, T., “Low-cost runge-kutta integrators for in-
		compressible flow simulations,” Journal of computational physics, vol. 443, no. 110518,
		p. 110518, 2021.
	'''
	# Initialization of variables
	N  = mesh['N'][0]
	M  = mesh['N'][1]
	
	Ku1, Ku2 = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	Kv1, Kv2 = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)
	usp, vsp = np.zeros((N+2,M+2),np.double), np.zeros((N+2,M+2),np.double)

	Ku1, Kv1 = u.copy(), v.copy()

	gpssx, gpssy = gradient(ps,mesh)
	# Loop the stages
	for ist in range(t['s']):
		# Evaluate R
		Rus, Rvs = calcR(Ku1,Kv1,nu,mesh)
	
		# Compute up of the stage
		Ku2 = halo_update(t['A'][ist]*Ku2 + dt*Rus - t['c'][ist]*gpssx)
		Kv2 = halo_update(t['A'][ist]*Kv2 + dt*Rvs - t['c'][ist]*gpssy)
		
		# Update K1
		Ku1 = Ku1 + t['B'][ist]*Ku2 
		Kv1 = Kv1 + t['B'][ist]*Kv2 

	psp1 = solve_poisson(Ku1,Kv1,mesh)
	gpsx, gpsy = gradient(psp1,mesh)
	# Compute the next step velocities
	unp1 = (Ku1 - gpsx)# u^(n+1) = up - grad(ps)
	vnp1 = (Kv1 - gpsy)# v^(n+1) = vp - grad(ps)
	
	return unp1, vnp1, psp1
	#return Ku1, Kv1, pss
def eigenCDRK(f,nu,u,v,gamma,mesh):
	'''Calculation of the optimum time step for a ERK method solving the problem
	'''
	# Compute bounds
	a, b = boundCD(u,v,nu,mesh)
	# Compute phi between [0,pi/2]
	phi  = np.abs(np.arctan(b/a)) # arctan computes between [-pi/2,pi/2]
	# Evaluate and obtain the region of stability
	mustbezero = lambda x : np.abs(np.polyval(gamma,-x*np.exp(phi*1j))) - 1
	Topt = fsolve(mustbezero,16.)[0]
	dt   = Topt/np.sqrt(a**2. + b**2.)
	# Return
	return f*dt, 1.0

def eigenCDRK_pol(f,nu,u,v,gamma,mesh):
    '''Calculation of the optimum time step for a ERK method via approximate ROS
    '''
    # Compute bounds
    a, b = boundCD(u,v,nu,mesh)
    # Compute phi between [0,pi/2]
    phi  = np.abs(np.arctan(b/a)) # arctan computes between [-pi/2,pi/2]
     # Evaluate and obtain the region of stability
    Topt = np.polyval(gamma,phi)
    dt   = Topt/np.sqrt(a**2. + b**2.)
    # Return
    return f*dt, 1.0

def fixed_dt(f,nu,u,v,gamma,mesh):
	return hardDt, 1.0

## PREPROCESS
mesh = create_mesh( L, N, M )  # creation of the uniform mesh
td   = 0.                      # actual time, set to zero
ti   = 0                       # iteration counter

# Different variables allocation for posterior plotting
uAnalytic = [] # analitical u evolution at (ip,jp) point
vAnalytic = [] # analytical v evolution ...
uNumeric  = [] # numerical  u evolution ...
vNumeric  = [] # numerical  v evolution ...

time      = [] # time corresponding to the saved values

# Initial conditions
u,v   = set_field( ufun, vfun, td, mesh ) # evaluate u and v at nodal positions
unm1  = halo_update(u.copy()) # u^(n-1) = u^(n)
vnm1  = halo_update(v.copy()) # v^(n-1) = v^(n)
ps    = np.zeros_like(u)
psm1  = np.zeros_like(u)
psm2  = np.zeros_like(u)
gamma = []


if forced_tstep:
	calcdt = fixed_dt 
elif eigen_tstep:
	calcdt = eigenCD 	
else:
	calcdt = CFLcondition


# Compute low storage RK tableau
if integration.lower() in ['rk','rungekutta']:
	tableau = LSERK_tableau[RK_method] if RK_method in LSERK_tableau.keys() else ERK_tableau[RK_method]
	is_2NRK = RK_method in LSERK_tableau.keys()

# Region of stability for eigenCD
if eigen_tstep and ('rk' in integration.lower() or 'rungekutta' in integration.lower()):
    if approximate_ROS:
        calcdt = eigenCDRK_pol
        plot_region_of_stability(convert(tableau) if is_2NRK else tableau)
        gamma = tableau['ROS']
    else:
        calcdt = eigenCDRK
        plot_region_of_stability(convert(tableau) if is_2NRK else tableau)
        gamma = compute_region_of_stability(convert(tableau) if is_2NRK else tableau)

for f in np.arange(fmin,fmax+0.1,0.1):
## MAIN LOOP
	while True:
		# evaluate the analytic solution at this instant
		usol, vsol = set_field( ufun, vfun, td, mesh ) # evaluate u and v at nodal positions

		# store the actual time solved
		time.append(td)

		# store the analytic and numeric solutions for every velocity
		uNumeric.append(  u[iref+1,jref+1]) # remember that ip and jp are ref to INNER DOMAIN
		vNumeric.append(  v[iref+1,jref+1])
		uAnalytic.append( usol[iref+1,jref+1])
		vAnalytic.append( vsol[iref+1,jref+1])

		# Numerical computations
		u = halo_update(u) # load the values of u to halos
		v = halo_update(v) #  ... of v to ...
		
		dt, beta = calcdt(f,nu,u,v,gamma,mesh) # dt = f*min( min(delta/|u|,0.5min(delta²/nu)

		print('Iteration %04d, time %.2f, dt %.2e, beta %.2f - '%(ti,td,dt,beta),end='')
		print('umax %.2f, umin %.2f, uavg %.2f - '%(np.min(u),np.max(u),np.mean(u)), end='')
		print('vmax %.2f, vmin %.2f, vavg %.2f'%(np.min(v),np.max(v),np.mean(v)))

		# Temporal integration scheme
		if integration.lower() in ['ab','adamsbashforth']:
			# Calculate R terms
			Ru, Rv     = calcR(u,v,nu,mesh)       # Ri(n) = -Ci+Di*nu
			# Recompute for the previous step - not efficient but well...
			Rum1, Rvm1 = calcR(unm1,vnm1,nu,mesh) # Ri(n) = -Ci+Di*nu

			# Calculare predictor velocity
			# Adams Bashforth method as up = un + dt(1.5R(n)-0.5R(n-1))
			up = halo_update(u + dt*(1.5*Ru - 0.5*Rum1))
			vp = halo_update(v + dt*(1.5*Rv - 0.5*Rvm1))

			# Solve the poisson equation to obtain the pseudo-pressure
			psp1 = solve_poisson(up,vp,mesh)

			# Obtain the gradient of the pseudo-pressure
			gpsx, gpsy = gradient(psp1,mesh)

			# Compute the next step velocities
			unp1 = up - gpsx # u^(n+1) = up - grad(ps)
			vnp1 = vp - gpsy # v^(n+1) = vp - grad(ps)

		if integration.lower() in ['lf','leapfrog']:
			# Compute ustar
			us = (1+beta)*u - beta*unm1
			vs = (1+beta)*v - beta*vnm1

			# Calculate R terms
			Ru, Rv = calcR(us,vs,nu,mesh)       # Ri(n) = -Ci+Di*nu

			# Calculare predictor velocity
			# Leapfrog method as up = un + dt*R((1+beta)*u(n)-beta*u(n-1))
			up = halo_update(2.*beta*u - (beta-0.5)*unm1 + dt*Ru)
			vp = halo_update(2.*beta*v - (beta-0.5)*vnm1 + dt*Rv)

			# Solve the poisson equation to obtain the pseudo-pressure
			psp1 = solve_poisson(up,vp,mesh)

			# Obtain the gradient of the pseudo-pressure
			gpsx, gpsy = gradient(psp1,mesh)

			# Compute the next step velocities
			unp1 = (up - gpsx)/(beta+0.5) # u^(n+1) = up - grad(ps)
			vnp1 = (vp - gpsy)/(beta+0.5) # v^(n+1) = vp - grad(ps)

		if integration.lower() in ['rk','rungekutta']:
			# Calculate R terms
			if is_2NRK:
				if approximation.lower() in ['consistent']: 
					unp1, vnp1, psp1 = calc_LSERK_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
				else:
					if ti < tableau['s']: # for the self starting capabilities
						unp1, vnp1, psp1 = calc_LSERK_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
					else:
						unp1, vnp1, psp1 = calc_LSERK_approximate(u,v,ps,psm1,approximation,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
				
			else:
				# Calculate R terms
				if approximation.lower() in ['consistent']: 
					if tableau['diagonal']:
						Ru, Rv = calcR_ERK_diagonal_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
					else:
						Ru, Rv = calcR_ERK_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
				else: 
					if approximation.lower() in ['approximated_1']:
						if ti == 0: # we only need pn for the aprox
							if tableau['diagonal']:
								Ru, Rv = calcR_ERK_diagonal_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
							else:
								Ru, Rv = calcR_ERK_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
						else:
							if tableau['diagonal']:
								Ru, Rv = calcR_ERK_diagonal_aproximate(u,v,ps,psm1,psm2,approximation,dt,nu,tableau,mesh)
							else:
								Ru, Rv = calcR_ERK_aproximate(u,v,ps,psm1,psm2,approximation,dt,nu,tableau,mesh)
					else:
						if ti < tableau['order']: # we may need pn, pnm1... for the aprox
							if tableau['diagonal']:
								Ru, Rv = calcR_ERK_diagonal_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
							else:
								Ru, Rv = calcR_ERK_consistent(u,v,dt,nu,tableau,mesh) # Ri(n) = -Ci+Di*nu
						else:
							if tableau['diagonal']:
								Ru, Rv = calcR_ERK_diagonal_aproximate(u,v,ps,psm1,psm2,approximation,dt,nu,tableau,mesh)
							else:
								Ru, Rv = calcR_ERK_aproximate(u,v,ps,psm1,psm2,approximation,dt,nu,tableau,mesh)
				# Calculare predictor velocity
				up = halo_update(u + dt*Ru)
				vp = halo_update(v + dt*Rv)
			
				# Solve the poisson equation to obtain the pseudo-pressure
				psp1 = solve_poisson(up,vp,mesh)

				# Obtain the gradient of the pseudo-pressure
				gpsx, gpsy = gradient(psp1,mesh)

				# Compute the next step velocities
				unp1 = up - gpsx # u^(n+1) = up - grad(ps)
				vnp1 = vp - gpsy # v^(n+1) = vp - grad(ps)
				#div = np.max(divergence(unp1,vnp1,mesh))
				#print("Div final =",div)
		

		# Advance time
		td += dt
		ti += 1

		psm2 = psm1.copy()
		unm1 = u.copy()    # u^(n-1) = u^(n)
		vnm1 = v.copy()    # v^(n-1) = v^(n)
		psm1 = ps.copy()
		u    = unp1.copy() # u^(n)   = u^(n+1)
		v    = vnp1.copy() # v^(n)   = v^(n+1)
		ps   = psp1.copy() # p^(n)   = p^(n+1)

		# check if the case has ended succesfully
		if td > tend:
			usol, vsol = set_field( ufun, vfun, td, mesh ) # evaluate u and v at nodal positions
			aux1 = [x**2 for x in np.subtract(u,usol)]
			aux2 = [x**2 for x in np.subtract(v,vsol)]
			eu = np.sqrt(np.sum(aux1))
			ev = np.sqrt(np.sum(aux2))
			L2norm = eu
			break

	# Print some metrics
	print('END')
	# end time
	print('tend: ',td-dt)
	# temporal scheme
	print('Temporal scheme: ',integration)
	if integration.lower() in ['rk','rungekutta']: print('RK scheme: ',RK_method,' ',approximation)
	# security factor f
	print('f: ',f)
	# number of iterations
	print('Number of iterations: ',ti)
	# mean dt
	print('Average dt: ',td/ti)
	
	uAna = np.array(uAnalytic)
	uNum = np.array(uNumeric)
	# One-point temporal MSE
	print('MSE: ',np.mean((uAna-uNum)**2))
	# One-point maximum dimensional error at end
	print('EAE:',max(np.abs(uNumeric[-1]-uAnalytic[-1]),np.abs(vNumeric[-1]-vAnalytic[-1])))
	# L2 norm of the error of the fields
	print('L2E:',L2norm)
	

	# RESTART
	td   = 0.                      # actual time, set to zero
	ti   = 0                       # iteration counter

	# Different variables allocation for posterior plotting
	uAnalytic = [] # analitical u evolution at (ip,jp) point
	vAnalytic = [] # analytical v evolution ...
	uNumeric  = [] # numerical  u evolution ...
	vNumeric  = [] # numerical  v evolution ...

	time      = [] # time corresponding to the saved values

	# Initial conditions
	u,v   = set_field( ufun, vfun, td, mesh ) # evaluate u and v at nodal positions
	unm1  = halo_update(u.copy()) # u^(n-1) = u^(n)
	vnm1  = halo_update(v.copy()) # v^(n-1) = v^(n)
	ps    = np.zeros_like(u)
	psm1  = np.zeros_like(u)
	psm2  = np.zeros_like(u)
# Plots
plt.figure()
plt.plot(time,uAnalytic,'--',label='u analytic')
plt.plot(time,vAnalytic,'--',label='v analytic')
plt.plot(time,uNumeric,label='u numeric')
plt.plot(time,vNumeric,label='v numeric')
plt.ylabel('Velocity [m/s]')
plt.grid(visible = True)
plt.xlabel('Time [s]')
plt.legend()

plt.figure()
plt.plot(time,np.abs(np.subtract(uNumeric,uAnalytic)/uAnalytic)*100,label='u error')
plt.plot(time,np.abs(np.subtract(vNumeric,vAnalytic)/vAnalytic)*100,label='v error')
plt.ylabel('Error (%)')
plt.xlabel('Time [s]')
plt.grid(visible = True)
plt.legend()

# plt.show()
