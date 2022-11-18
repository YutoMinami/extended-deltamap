import os
import sympy 
import scipy 
import healpy 
import numpy
from scipy.special import sph_harm, factorial, lpmv,perm
# https://arxiv.org/abs/astro-ph/0508514

class Covariance:
	def __init__(self, nside = 4, lmax =None, maskname=None, pixwin=True, fwhm=None,verbose = 0):
		"""
		"""
		self.nside = nside
		self.npix = healpy.nside2npix(nside)
		if lmax is None:
			self.lmax = 2*nside
		elif lmax > self.nside*4-1:
			self.lmax = self.nside*4-1
		else:
			self.lmax = lmax
		self.lmin = 2
		tmpdir = os.path.dirname(__file__)
		self.cl_scalar_file = tmpdir + '/files/test_lensedcls_49T7H5WT3X.dat'
		self.cl_tensor_file = tmpdir + '/files/test_tenscls_49T7H5WT3X.dat'
		self.maskname = maskname
		self.pixwin = pixwin
		self.fwhm = fwhm
		self.indices = None
		self.size = None
		self.angles = None
		self.verbose = verbose
		self.m_indices = None
		self.cl_scalar = None
		self.cl_tensor = None
		self.mvec = None
		self.lvec = None
		self.Warray = None
		self.Xarray = None

	def ReturnBell(self,ell, fwhm):
		s=2.
		sigma_b =( fwhm * numpy.pi/10800.)/numpy.sqrt(8.*numpy.log(2))
		return numpy.exp(-(ell*(ell+1)-s**2)*pow(sigma_b,2)/2)	
	def CreateUnmaskedIndices(self):
		if not self.maskname is None:
			self.mask = healpy.read_map(self.maskname, field=(0),verbose = self.verbose)
			self.mask =healpy.ud_grade(self.mask, nside_out= self.nside)
			self.indices = numpy.where(self.mask != 0.0 )[0]
			self.size = len(self.indices)
		else:
			self.indices = numpy.arange(self.npix)
			self.size = len(self.indices)

	def CreateAngles(self):
		self.angles = numpy.asarray(healpy.pix2ang( self.nside ,  self.indices)).transpose()
		self.m_indices = numpy.arange(len(self.angles))

	def Initialise(self):
		self.CreateUnmaskedIndices()
		self.CreateAngles()
		self.SetCl()
		#self.CalcWi()
		#self.CalcXi()
		self.PrepareVec()
		self.CalcWandXarray()

	def SetClFiles(self, lensedcl, tensorcl):
		self.cl_scalar_file = lensedcl
		self.cl_tensor_file = tensorcl
		
	def ReturnAlphalm(self, ell, m, theta, plus=True):
		sgn = -1. if plus else 1.0
		alphalm = ( 2*pow(m,2) - ell*(ell+1))/pow(numpy.sin(theta) ,2)
		alphalm += sgn* 2.*m*(ell-1)/ numpy.tan(theta) /numpy.sin(theta)
		alphalm += ell*(ell-1)/pow(numpy.tan(theta),2)
		return alphalm

	def ReturnBetalm(self, ell , m , theta, plus=True):
		sgn = 1. if plus else -1.0
		betalm = 2.*numpy.sqrt(
			(2.*ell+1)/(2.*ell-1)*( pow(ell,2) - pow(m,2))
		)*(
			sgn*m/pow(numpy.sin(theta) ,2) + 1./numpy.tan(theta)/numpy.sin(theta) 
		)	
		return betalm
	
	def ReturnNlmArray(self):
		consts = numpy.full(len(self.lvec), 4)
		nlm = numpy.full( len(self.lvec),1./perm(self.lvec+2, consts , exact=False) )
		mask = self.mvec >0
		nlm[mask] /= perm( self.lvec[mask]+self.mvec[mask], 2*self.mvec[mask], exact=False  )
		mask = self.mvec < 0
		nlm[mask] *= perm( self.lvec[mask] - self.mvec[mask], -2*self.mvec[mask], exact=False  )
		return 2.*numpy.sqrt(nlm)


	def ReturnNlm(self,ell, m):
		value=1./perm(ell+2,4, exact=True)
		if m >0:
			value /=perm(ell+m,2*m, exact=True)
		elif m <0:
			value *=perm(ell-m,-2*m, exact=True)
		else:
			pass
		return 2.*numpy.sqrt(
		#factorial( ell-2, True)*factorial(ell-m, True)/factorial( ell + 2, True)/factorial(ell+m, True) 
		value
		)
	def ReturnF1lmArray(self,thetas, Plms_first, Plms_second):
		first = -1.0 * ( 
			(self.lvec- pow(self.mvec,2))/pow(numpy.sin(thetas),2) + self.lvec*(self.lvec-1)/2. 
			) * Plms_first
		second = (self.lvec + self.mvec) * numpy.cos(thetas)/pow( numpy.sin(thetas),2) * Plms_second
		return first+second

	def ReturnF1lm(self, ell, m ,theta ):
		first = -1.0 * ( (ell- pow(m,2))/pow(numpy.sin(theta),2) + ell*(ell-1)/2. ) *lpmv( int(m), int(ell), numpy.cos(theta))
		#print(lpmv( int(m), int(ell), numpy.cos(theta)))
		second = (ell+m) * numpy.cos(theta)/pow( numpy.sin(theta),2) *lpmv(int(m), int(ell-1), numpy.cos(theta)) if abs(m)<= (ell-1) else 0.0

		return self.ReturnNlm(int(ell), int(m))*	(first+second)

	def ReturnF2lmArray(self,thetas, Plms_first, Plms_second):
		first = -1.0 *( self.lvec-1 )*numpy.cos(thetas) * Plms_first
		second = (self.lvec+self.mvec) * Plms_second
		coeff = self.mvec/pow(numpy.sin(thetas),2) 
		return 	coeff*(first+second)

	def ReturnF2lm(self, ell, m ,theta ):
		first = -1.0 *( ell-1 )*numpy.cos(theta) *lpmv( int(m),int( ell), numpy.cos(theta))
		second = (ell+m) * lpmv(int(m), int(ell-1), numpy.cos(theta)) if abs(m)<= (ell-1) else 0.0
		return self.ReturnNlm(int(ell), int(m)) *	m/pow(numpy.sin(theta),2)	*(first+second)

	def ReturnWorX(self, ell , m , theta, phi, isW = True):	
		if abs(m) > ell:
			raise Exception('|m| should be <= ell')
		if ell <= 1:
			raise Exception('ell should be >= 2')
		ell = numpy.float64(ell)
		m = numpy.float64(m)
		coeff = numpy.sqrt( (2*ell+1)/4/numpy.pi)*numpy.exp(1.0j*m * phi )
		#KaminovSki
		#coeff = numpy.sqrt( 2* (2*ell+1)/4/numpy.pi)*numpy.exp(1.0j*m * phi )
		if isW:
			return 1.0*coeff *  self.ReturnF1lm( ell, m , theta)
		else:
			return 1.0j*coeff *  self.ReturnF2lm( ell, m , theta)

	def Return2Ylm(self, ell , m , theta, phi, plus = True):	
		if abs(m) > ell:
			raise Exception('|m| should be <= ell')
		if ell <= 1:
			raise Exception('ell should be >= 2')
		ell = numpy.float64(ell)
		m = numpy.float64(m)
		alphalm = self.ReturnAlphalm(ell, m, theta, plus)
		betalm = self.ReturnBetalm(ell, m, theta, plus)
		aterm = 	alphalm * sph_harm( m, ell,phi, theta)
		bterm = 0.0 if (ell-1) < abs(m) else betalm * sph_harm(m, ell-1,phi, theta)
		if self.verbose >= 2:
			print('l m', ell, m,'Yl,m Yl-1,m' ,sph_harm( m, ell, phi,theta) ,sph_harm( m, ell-1, phi ,theta),'alphalm : ',alphalm ,'betalm : ',betalm ,'aterm : ',aterm ,'bterm : ',bterm  )
		return numpy.sqrt(  
			factorial(ell-2, True) / factorial(ell + 2,True) 
		)*(aterm + bterm)


	def ReturnWlm(self, ell, m, theta, phi):
		#return numpy.sqrt(2.)*(self.Return2Ylm(ell, m, theta, phi, plus = True) + self.Return2Ylm(ell, m, theta, phi, plus = False ) )/2.
		return 	self.ReturnWorX( ell, m, theta , phi, True)

	def ReturnXlm(self, ell, m, theta, phi):
		#return numpy.sqrt(2.)*1.0j*(self.Return2Ylm(ell, m, theta, phi, plus = True) - self.Return2Ylm(ell, m, theta, phi, plus = False ) )/2.
		return 	self.ReturnWorX( ell, m, theta , phi, False)
	

	def ReturnWi(self, index):
		angle = (self.angles[index][0], self.angles[index][1] )
		Wi = []
		for ell in range(2,self.lmax+1,1):
			Wl = []
			for m in range(-ell, ell+1, 1):
				Wl.append( self.ReturnWlm(ell, m, angle[0], angle[1]) )
			Wi.append(Wl)
		return Wi

	def ReturnXi(self, index):
		angle = (self.angles[index][0], self.angles[index][1] )
		Xi = []
		for ell in range(2,self.lmax+1,1):
			Xl = []
			for m in range(-ell, ell+1, 1):
				Xl.append( self.ReturnXlm(ell, m, angle[0], angle[1]) )
			Xi.append(Xl)
		return Xi

	def PrepareVec(self):
		self.lvec = numpy.zeros( 
			int( (self.lmax-self.lmin+1 )*(self.lmax+self.lmin+1)   ) 
		)
		self.mvec = numpy.copy(self.lvec)

		for ell in range( self.lmin, self.lmax+1,1):
			ms = numpy.arange( -ell , ell+1,1)
			ls = numpy.full( 2*ell+1  ,ell)
			self.lvec[ (ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  + len(ls) ] = ls
			self.mvec[ (ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  + len(ms) ] = ms
		if self.verbose:
			print( ' lvec ', self.lvec)
			print( ' mvec ', self.mvec)

	def CalcWandXarray(self):
		self.Warray = numpy.zeros((len(self.m_indices), len(self.lvec) ), dtype='complex128')
		self.Xarray = numpy.zeros((len(self.m_indices), len(self.lvec) ), dtype='complex128')
		self.Nlmarray = self.ReturnNlmArray()
		mask = (abs(self.mvec) <= (self.lvec -1) )
		for idx in  self.m_indices:
			thetas = numpy.full( len(self.lvec)  , self.angles[idx][0] )
			phis = numpy.full( len(self.lvec)  , self.angles[idx][1] )
			Plms_first = numpy.zeros( len(self.lvec) )

			Plms_second = numpy.zeros( len(self.lvec) )

			Plms_second[mask] = lpmv(
					self.mvec[mask].astype(numpy.int64), self.lvec[mask].astype(numpy.int64)-1, numpy.cos(thetas[mask])
				)
			Plms_first = lpmv(
					self.mvec.astype(numpy.int64), self.lvec.astype(numpy.int64), numpy.cos(thetas)
				)
			F1lmarray = self.ReturnF1lmArray(thetas, Plms_first, Plms_second)
			F2lmarray = self.ReturnF2lmArray(thetas, Plms_first, Plms_second)
			coeff = numpy.sqrt( (2*self.lvec+1)/4./numpy.pi)*numpy.exp(1.0j*self.mvec * phis )
			self.Warray[idx] = 1.0 * coeff * self.Nlmarray *F1lmarray
			self.Xarray[idx] = 1.0j * coeff * self.Nlmarray *F2lmarray

		
	def CalcWi(self):
		self.Wi = [ self.ReturnWi(idx)  for idx in self.m_indices]
		#self.ReturnWi(self.m_indices) 

	def CalcXi(self):
		self.Xi = [ self.ReturnXi(idx)  for idx in self.m_indices]
		#self.Xi =self.ReturnXi(self.m_indices) 

	#read_cls
	def ReadCell(self, cl_s_name, nside, isScalar =True):
		cl_s = numpy.loadtxt(cl_s_name)
		if len(cl_s[0]) in (4, 6) and isScalar: 
			cl_s = numpy.c_[cl_s[:,:3], numpy.zeros(len(cl_s))[:,numpy.newaxis],cl_s[:,3]]
		cls, ls = cl_s.T[1:5], cl_s.T[0]
		cl = cls * 2.*numpy.pi/(ls*(ls+1.))
		#cl = numpy.c_[numpy.zeros([cl.shape[0],2]),cl]
		pw = healpy.pixwin(nside=nside , pol=True )
		lmax = min(min(len(cl[1]),len(cl[2])), len(pw[1][2:]))
		cl[1][:lmax] = cl[1][:lmax]*pow(pw[1][2:lmax+2],2)
		cl[2][:lmax] = cl[2][:lmax]*pow(pw[1][2:lmax+2],2)

		return cl	
	def SetCl(self):
		self.cl_scalar = self.ReadCell(self.cl_scalar_file, self.nside, True)
		self.cl_tensor = self.ReadCell(self.cl_tensor_file, self.nside, False)

	def ReturnEllSum(self, YY):
		YYl = numpy.zeros( (YY.shape[0], YY.shape[1], self.lmax-self.lmin+1) , dtype =numpy.complex128)
		for idx,ell in enumerate(range( self.lmin, self.lmax+1,1) ):
			YYl[:,:,idx] = numpy.sum( YY[:,:,(ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ], axis=2)
		return YYl

	def ReturnWWlArray(self, i ,j):
		WWarray =self.Warray[i] * numpy.conjugate(self.Warray[j])

		WWl = numpy.zeros(  self.lmax-self.lmin+1 , dtype =numpy.complex128)
		for idx,ell in enumerate(range( self.lmin, self.lmax+1,1) ):
			WWl[idx] = sum(WWarray[ (ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ])
		return WWl

	def ReturnXXlArray(self, i ,j):
		XXarray =self.Xarray[i] * numpy.conjugate(self.Xarray[j])

		XXl = numpy.zeros(  self.lmax-self.lmin+1 , dtype =numpy.complex128)
		for idx,ell in enumerate(range( self.lmin, self.lmax+1,1) ):
			XXl[idx] = sum(XXarray[ (ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ])
		return XXl

	def ReturnWXlArray(self, i ,j):
		WXarray =self.Warray[i] * numpy.conjugate(self.Xarray[j])

		WXl = numpy.zeros(  self.lmax-self.lmin+1 , dtype =numpy.complex128)
		for idx,ell in enumerate(range( self.lmin, self.lmax+1,1) ):
			WXl[idx] = sum(WXarray[ (ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ])
		return WXl

	def ReturnXWlArray(self, i ,j):
		XWarray =self.Xarray[i] * numpy.conjugate(self.Warray[j])

		XWl = numpy.zeros(  self.lmax-self.lmin+1 , dtype =numpy.complex128)
		for idx,ell in enumerate(range( self.lmin, self.lmax+1,1) ):
			XWl[idx] = sum(XWarray[ (ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ])
		return XWl


	def ReturnWWl(self, i,j ):
		Wi = self.Wi[i]
		Wj = self.Wi[j]
		#WWl = [ numpy.real( sum(Wil *  numpy.conjugate(Wjl)  )  ) for Wil,Wjl in zip(Wi,Wj)  ]
		WWl = [ ( sum(Wil *  numpy.conjugate(Wjl)  )  ) for Wil,Wjl in zip(Wi,Wj)  ]
		return WWl
	def ReturnXXl(self, i,j ):
		Xi = self.Xi[i]
		Xj = self.Xi[j]
		#XXl = [ numpy.real( sum(Xil *  numpy.conjugate(Xjl)  )      ) for Xil,Xjl in zip(Xi,Xj)  ]
		XXl = [ sum(Xil *  numpy.conjugate(Xjl)  )  for Xil,Xjl in zip(Xi,Xj)  ]
		return XXl
	def ReturnWXl(self, i,j ):
		Wi = self.Wi[i]
		Xj = self.Xi[j]
		#WXl = [ numpy.real( sum(Wil *  numpy.conjugate(Xjl)  )      ) for Wil,Xjl in zip(Wi,Xj)  ]
		WXl = [ sum(Wil *  numpy.conjugate(Xjl)  )  for Wil,Xjl in zip(Wi,Xj)  ]
		return WXl

	def ReturnXWl(self, i,j ):
		Xi = self.Xi[i]
		Wj = self.Wi[j]
		#XWl = [ numpy.real( sum(Xil *  numpy.conjugate(Wjl)  )      ) for Xil,Wjl in zip(Xi,Wj)  ]
		XWl = [ sum(Xil *  numpy.conjugate(Wjl)  )  for Xil,Wjl in zip(Xi,Wj)  ]
		return XWl
	
	def CalcCovArray(self,Cl):
		self.QQ = numpy.zeros( shape=(self.size, self.size ),dtype='float64' )
		self.UU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )
		self.QU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )
		ell = numpy.arange(2, self.lmax+1 )
		pw_l = numpy.ones(len(ell))
		if not self.fwhm is None:
			pw_l = self.ReturnBell(ell, self.fwhm)

		WWl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		XXl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		WXl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		XWl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		for idx,ell in enumerate(range( self.lmin, self.lmax+1,1) ):
			W = self.Warray[:,(ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ]
			X = self.Xarray[:,(ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ]
			WW = numpy.einsum('i...,j...->ij...', W,numpy.conjugate(W) )
			XX = numpy.einsum('i...,j...->ij...', X,numpy.conjugate(X) )
			WX = numpy.einsum('i...,j...->ij...', W,numpy.conjugate(X) )
			XW = numpy.einsum('i...,j...->ij...', X,numpy.conjugate(W) )

			WWl[:,:,idx] =  numpy.real( numpy.sum(WW, axis =2) )
			XXl[:,:,idx] =  numpy.real( numpy.sum(XX, axis =2)  )
			WXl[:,:,idx] =  numpy.real( numpy.sum(WX, axis =2)  )
			XWl[:,:,idx] =  numpy.real( numpy.sum(XW, axis =2)  )

		print(WWl.shape[2])
		self.QQ = numpy.sum(WWl[...,:] * Cl[1][:WWl.shape[2]]*pow(pw_l,2) , axis=2) ##EE
		self.QQ += numpy.sum(XXl[...,:] * Cl[2][:XXl.shape[2]]*pow(pw_l,2) , axis=2) ##BB 
		#self.QQ[i,j] -= sum(XXl * Cl[2][:len(XXl)] ) ##BB 
		#self.QQ = self.QQ[i,j] ## TODO
		##EB is not yet implemented

		self.UU = numpy.sum(XXl[...,:]  * Cl[1][:XXl.shape[2]]*pow(pw_l,2), axis = 2) ##EE
		#self.UU[i,j] = -sum(XXl * Cl[1][:len(XXl)]) ##EE
		self.UU += numpy.sum( WWl[...,:]  * Cl[2][:WWl.shape[2]]*pow(pw_l,2) ,axis=2) ##BB a
		#self.UU[j,i] = self.UU[i,j] ##TODO

		self.QU = -1.* numpy.sum( WXl[...,:]  * Cl[1][:WXl.shape[2]]*pow(pw_l,2) , axis=2) ##EE
		self.QU += numpy.sum( XWl[...,:]  * Cl[2][:XWl.shape[2]]*pow(pw_l,2), axis=2) ##BB 
		self.QU = -1.*numpy.sum( XWl[...,:] * Cl[1][:XWl.shape[2]]*pow(pw_l,2), axis=2) ##EE
		self.QU += numpy.sum( WXl[...,:]  * Cl[2][:WXl.shape[2]]*pow(pw_l,2), axis=2) ##BB 
	"""
	def CalcCovArray(self,Cl):
		self.QQ = numpy.zeros( shape=(self.size, self.size ),dtype='float64' )
		self.UU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )
		self.QU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )
		ell = numpy.arange(2, self.lmax+1 )
		pw_l = numpy.ones(len(ell))
		if not self.fwhm is None:
			pw_l = self.ReturnBell(ell, self.fwhm)

		WWl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		XXl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		WXl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		XWl = numpy.zeros( shape=(self.size, self.size, self.lmax-self.lmin+1),dtype='float64' )
		for idx,ell in enumerate(range( self.lmin, self.lmax+1,1) ):
			W = self.Warray[:,(ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ]
			X = self.Xarray[:,(ell - self.lmin )*(ell + self.lmin) : (ell - self.lmin )*(ell + self.lmin)  +  2*ell+1 ]
			WW = numpy.einsum('i...,j...->ij...', W,numpy.conjugate(W) )
			XX = numpy.einsum('i...,j...->ij...', X,numpy.conjugate(X) )
			WX = numpy.einsum('i...,j...->ij...', W,numpy.conjugate(X) )
			XW = numpy.einsum('i...,j...->ij...', X,numpy.conjugate(W) )

			WWl[:,:,idx] =  numpy.real( numpy.sum(WW, axis =2) )
			XXl[:,:,idx] =  numpy.real( numpy.sum(XX, axis =2)  )
			WXl[:,:,idx] =  numpy.real( numpy.sum(WX, axis =2)  )
			XWl[:,:,idx] =  numpy.real( numpy.sum(XW, axis =2)  )

		print(WWl.shape[2])
		self.QQ = numpy.sum(WWl[...,:] * Cl[1][:WWl.shape[2]]*pow(pw_l,2) , axis=2) ##EE
		self.QQ += numpy.sum(XXl[...,:] * Cl[2][:XXl.shape[2]]*pow(pw_l,2) , axis=2) ##BB 
		#self.QQ[i,j] -= sum(XXl * Cl[2][:len(XXl)] ) ##BB 
		#self.QQ = self.QQ[i,j] ## TODO
		##EB is not yet implemented

		self.UU = numpy.sum(XXl[...,:]  * Cl[1][:XXl.shape[2]]*pow(pw_l,2), axis = 2) ##EE
		#self.UU[i,j] = -sum(XXl * Cl[1][:len(XXl)]) ##EE
		self.UU += numpy.sum( WWl[...,:]  * Cl[2][:WWl.shape[2]]*pow(pw_l,2) ,axis=2) ##BB a
		#self.UU[j,i] = self.UU[i,j] ##TODO

		self.QU = -1.* numpy.sum( WXl[...,:]  * Cl[1][:WXl.shape[2]]*pow(pw_l,2) , axis=2) ##EE
		self.QU += numpy.sum( XWl[...,:]  * Cl[2][:XWl.shape[2]]*pow(pw_l,2), axis=2) ##BB 
		self.QU = -1.*numpy.sum( XWl[...,:] * Cl[1][:XWl.shape[2]]*pow(pw_l,2), axis=2) ##EE
		self.QU += numpy.sum( WXl[...,:]  * Cl[2][:WXl.shape[2]]*pow(pw_l,2), axis=2) ##BB 
	"""
	def CalcCov(self, Cl):
		self.QQ = numpy.zeros( shape=(self.size, self.size ),dtype='float64' )
		self.UU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )
		self.QU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )
		ell = numpy.arange(2, self.lmax+1 )
		pw_l = numpy.ones(len(ell))
		if self.pixwin :
			pw_l = healpy.pixwin(self.nside , pol=True, lmax=self.lmax)[1][2:]
		if not self.fwhm is None:
			pw_l = self.ReturnBell(ell, self.fwhm)
			#healpy.pixwin(self.nside , pol=True, lmax=self.lmax)[1][2:]

		for i in range(self.size):
			for j in range(self.size):
				if j > i :
					continue
				WWl = numpy.real(self.ReturnWWlArray( i,j) )*pow(pw_l,2)
				XXl = numpy.real( self.ReturnXXlArray( i,j)  )*pow(pw_l,2)
				WXl =  numpy.real(self.ReturnWXlArray( i,j) )*pow(pw_l,2)
				XWl =  numpy.real(self.ReturnXWlArray( i,j) )*pow(pw_l,2)
				WWl = numpy.real(self.ReturnWWl( i,j) )*pow(pw_l,2)
				XXl = numpy.real( self.ReturnXXl( i,j)  )*pow(pw_l,2)
				WXl =  numpy.real(self.ReturnWXl( i,j) )*pow(pw_l,2)
				XWl =  numpy.real(self.ReturnXWl( i,j) )*pow(pw_l,2)
				self.QQ[i,j] = sum(WWl * Cl[1][:len(WWl)] ) ##EE
				self.QQ[i,j] += sum(XXl * Cl[2][:len(XXl)] ) ##BB 
				#self.QQ[i,j] -= sum(XXl * Cl[2][:len(XXl)] ) ##BB 
				self.QQ[j,i] = self.QQ[i,j]
				##EB is not yet implemented

				self.UU[i,j] = sum(XXl * Cl[1][:len(XXl)]) ##EE
				#self.UU[i,j] = -sum(XXl * Cl[1][:len(XXl)]) ##EE
				self.UU[i,j] += sum(WWl * Cl[2][:len(WWl)]) ##BB a
				self.UU[j,i] = self.UU[i,j]

				self.QU[i,j] = -1.*sum( WXl * Cl[1][:len(WXl)]) ##EE
				self.QU[i,j] += sum( XWl * Cl[2][:len(XWl)]) ##BB 
				self.QU[j,i] = -1.*sum( XWl * Cl[1][:len(XWl)]) ##EE
				self.QU[j,i] += sum( WXl * Cl[2][:len(WXl)]) ##BB 
		#self.QQ = sympy.Matrix( self.QQ )
		#self.UU = sympy.Matrix( self.UU )
		#self.QU = sympy.Matrix( self.QU )

	def CalcCov02(self, Cl):
		self.QQ = numpy.zeros( shape=(self.size, self.size ),dtype='float64' )
		self.UU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )
		self.QU = numpy.zeros( shape=(self.size, self.size ),dtype='float64'  )

		ell = numpy.arange(2, self.lmax+1 )
		pw_l = numpy.ones(len(ell))

		if not self.fwhm is None:
			pw_l = self.ReturnBell(ell, self.fwhm)

		for i in range( self.size ):
			WW = numpy.einsum('i...,j...->ij...', self.Warray[i:i+1,:], numpy.conjugate(self.Warray[i:,:] ) )
			XX = numpy.einsum('i...,j...->ij...', self.Xarray[i:i+1,:], numpy.conjugate(self.Xarray[i:,:] ) )
			WX = numpy.einsum('i...,j...->ij...', self.Warray[i:i+1,:], numpy.conjugate(self.Xarray[:,:] ) )
			XW = numpy.einsum('i...,j...->ij...', self.Xarray[i:i+1,:], numpy.conjugate(self.Warray[:,:] ) )

			WWl = numpy.real(self.ReturnEllSum(WW) )
			XXl = numpy.real(self.ReturnEllSum(XX) )
			WXl = numpy.real(self.ReturnEllSum(WX) )
			XWl = numpy.real(self.ReturnEllSum(XW) )
			self.QQ[i,i:] =  numpy.sum( WWl[...,:] * Cl[1][:WWl.shape[2] ] * pow(pw_l,2), axis=2)[0] ##EE
			self.QQ[i,i:] += numpy.sum( XXl[...,:] * Cl[2][:XXl.shape[2] ] * pow(pw_l,2), axis=2)[0] ##BB 
			##EB is not yet implemented

			self.UU[i,i:] =  numpy.sum(XXl[...,:] * Cl[1][:XXl.shape[2]]*pow(pw_l,2) , axis=2)[0] ##EE
			self.UU[i,i:] += numpy.sum(WWl[...,:] * Cl[2][:WWl.shape[2]]*pow(pw_l,2) , axis=2)[0] ##BB 

			self.QU[i,:] = -1.*numpy.sum( WXl[...,:] * Cl[1][:WXl.shape[2]]*pow(pw_l,2) , axis=2)[0] ##EE
			self.QU[i,:] += numpy.sum( XWl[...,:] * Cl[2][:XWl.shape[2]]*pow(pw_l,2) , axis=2)[0] ##BB 
			#self.QU[i+1:,i] = -1.*numpy.sum( XWl[...,:] * Cl[1][:XWl.shape[2]]*pow(pw_l,2) , axis=2)[0][1:] ##EE
			#self.QU[i+1:,i] += numpy.sum( WXl[...,:] * Cl[2][:WXl.shape[2]]*pow(pw_l,2) , axis=2)[0][1:] ##BB 
		self.QQ = self.QQ + self.QQ.T - numpy.diag(self.QQ.diagonal())
		self.UU = self.UU + self.UU.T - numpy.diag(self.UU.diagonal())
		#self.QU = self.QU + numpy.conjugate(self.QU.T) - numpy.diag(self.QU.diagonal())

	def ReturnCovMatrix(self, scalar = True):
		if scalar:
			self.CalcCov02(self.cl_scalar)	
			#self.CalcCov(self.cl_scalar)	
			#self.CalcCovArray(self.cl_scalar)	
		else :
			self.CalcCov02(self.cl_tensor)	
			#self.CalcCov(self.cl_tensor)	
			#self.CalcCovArray(self.cl_tensor)	
		return numpy.block([[self.QQ, self.QU], [self.QU.T, self.UU] ])
		"""
		return  sympy.Matrix(
			sympy.BlockMatrix(
				[[self.QQ, self.QU], [self.QU.T, self.UU] ] 
			)
		)
		"""
