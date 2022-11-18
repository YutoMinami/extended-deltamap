import sympy
import numpy 
#from astropy.modeling.blackbody import blackbody_nu as B
from astropy import units
import scipy.constants.constants as constants
class Templates:
	def __init__(self):
		self.__models = []
		#['dust_MBB1','dust_MBB1', 'dust_MBB1']
		pass
	"""
	def ChangeNuRef(nu):
		self.nuRef = nu
	"""
	def PrintAll(self):
		pass

	def ReturnFunc(self):
		pass
	def ReturnTestPower(self, nuRef = 1.0):
		nu = sympy.Symbol('nu')
		beta_d = sympy.Symbol('beta_d')
		f = pow(nu/nuRef, beta_d ) 
		return f
	def ReturnPowerLawSynch_Norm(self, nuRef = 30.0):
		"""
		Returns
		Sympy function of one component modified black body
		"""
		sigma_beta_s = 0.0632
		nu = sympy.Symbol('nu')
		beta_s = sympy.Symbol('beta_s') * sigma_beta_s
 
		gnu = self.Return_gnu()
		f = sympy.Pow(nu/nuRef, beta_s ) *gnu
		return f

	def ReturnPowerLawSynch(self, nuRef = 30.0):
		"""
		Returns
		Sympy function of one component modified black body
		"""
		nu = sympy.Symbol('nu')
		beta_s = sympy.Symbol('beta_s')
 
		gnu = self.Return_gnu()

		f = sympy.Pow(nu/nuRef, beta_s ) *gnu
		return f
	def Return_gnu(self):
		T_CMB = 2.725

		nu = sympy.Symbol('nu')
		x_CMB =constants.h*nu*1.0e9/constants.k/T_CMB 
		gnu = sympy.Pow(sympy.exp(x_CMB) -1,2)/(x_CMB**2 * sympy.exp(x_CMB))
		return gnu

	def ReturnPowerLawDust(self, nuRef = 353.0):
		"""
		Returns 
		Sympy function of one component modified black body
		"""
		nu = sympy.Symbol('nu')
		beta_d = sympy.Symbol('beta_d')

		gnu = self.Return_gnu()
		f = gnu  * pow(nu/nuRef, beta_d ) 
		return f

	def ReturnMBB1_Norm(self, nuRef =  353.0):
		"""
		Returns 
		Sympy function of one component modified black body
		"""
		sigma_T_d = 2.254
		sigma_beta_d = 0.0542
		nu = sympy.Symbol('nu')
		beta_d = sympy.Symbol('beta_d') * sigma_beta_d
		T_d = sympy.Symbol('T_d1')*sigma_T_d

		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h*nuRef*1.0e9/constants.k/T_d
		gnu = self.Return_gnu()
		f = gnu * sympy.Pow(nu/nuRef, beta_d +1) * (sympy.exp(xRef)-1.)/(sympy.exp(x) -1. )
		return f

	def ReturnMBB1_xRef(self, nuRef =  353.0):
		"""
		Returns 
		Sympy function of one component modified black body
		"""
		nu = sympy.Symbol('nu')
		beta_d = sympy.Symbol('beta_d')
		xRef = sympy.Symbol('x^R') 
		#T_d = sympy.Symbol('T_d1')
		#x = constants.h*nu*1.0e9/constants.k/T_d
		x = xRef * (nu/nuRef)
		#xRef = constants.h*nuRef*1.0e9/constants.k/T_d
		gnu = self.Return_gnu()
		f = gnu * sympy.Pow(nu/nuRef, beta_d +1) * (sympy.exp(xRef)-1.)/(sympy.exp(x) -1. )
		return f

	def ReturnMBB1_xRef_Norm(self, nuRef =  353.0):
		"""
		Returns 
		Sympy function of one component modified black body
		"""
		#sigma_T_d = 2.254
		sigma_xRef = 0.09034
		sigma_beta_d = 0.0542
		nu = sympy.Symbol('nu')
		beta_d = sympy.Symbol('beta_d') * sigma_beta_d
		xRef = sympy.Symbol('x^R') * sigma_xRef
		#T_d = sympy.Symbol('T_d1')
		#x = constants.h*nu*1.0e9/constants.k/T_d
		x = xRef * (nu/nuRef)
		#xRef = constants.h*nuRef*1.0e9/constants.k/T_d
		gnu = self.Return_gnu()
		f = gnu * sympy.Pow(nu/nuRef, beta_d +1) * (sympy.exp(xRef)-1.)/(sympy.exp(x) -1. )
		return f

	def ReturnMBB1(self, nuRef =  353.0):
		"""
		Returns 
		Sympy function of one component modified black body
		"""
		nu = sympy.Symbol('nu')
		beta_d = sympy.Symbol('beta_d')
		T_d = sympy.Symbol('T_d1')
		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h*nuRef*1.0e9/constants.k/T_d
		gnu = self.Return_gnu()
		f = gnu * sympy.Pow(nu/nuRef, beta_d +1) * (sympy.exp(xRef)-1.)/(sympy.exp(x) -1. )
		return f
	"""
	def ReturnSync1(self, nuRef =  353.0):
	"""
