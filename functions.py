import numpy
from astropy.modeling.blackbody import blackbody_nu as B
from astropy import units
import scipy.constants.constants as constants
class Functions:
	def __init__(self):
		self.__models = ['dust_MBB1', 'synch_PL']
		self.nuRef_dust = 353.0
		self.nuRef_synch = 30.0
		self.sigma_T_d1 = 2.254
		self.sigma_beta_d1 = 0.0542
		self.sigma_beta_s = 0.0632
		self.T_CMB = 2.725
	
	def PrintModels(self):
		print(self.__models)

	def ReturnFunc(self, name):
		if not name in self.models:
			raise Exception('No such model with name of '+name +'!!!')
		else:
			for model in self.__models:
				break


	def Return_gnu(self, nu):
		x_CMB =constants.h*nu*1.0e9/constants.k/self.T_CMB 
		gnu = pow(numpy.expm1(x_CMB), 2 ) / ( pow(x_CMB,2) * numpy.exp(x_CMB))
		return gnu

	def ReturnMBB1(self, nu, mbeta_d1, mT_d1, betas_d1, Ts_d1):
		beta_d = (mbeta_d1 + betas_d1) * self.sigma_beta_d1
		T_d = (mT_d1 + Ts_d1) * self.sigma_T_d1
		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h * self.nuRef_dust * 1.0e9/constants.k / T_d
		gnu = self.Return_gnu(nu)	
		f = gnu * pow( nu / self.nuRef_dust , beta_d +1) * numpy.expm1(xRef)/numpy.expm1(x)
		return f

	def ReturnMBB1_DiffBeta(self, nu, mbeta_d1, mT_d1, betas_d1, Ts_d1):
		f = self.ReturnMBB1(nu, mbeta_d1, mT_d1, betas_d1, Ts_d1) * numpy.log( nu  / self.nuRef_dust )*self.sigma_beta_d1
		return f

	def ReturnMBB1_DiffT(self, nu, mbeta_d1, mT_d1, betas_d1, Ts_d1):
		beta_d = (mbeta_d1 + betas_d1) * self.sigma_beta_d1
		T_d = (mT_d1 + Ts_d1) * self.sigma_T_d1
		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h * self.nuRef_dust * 1.0e9/constants.k / T_d
		first= x*numpy.exp(x)/numpy.expm1(x)
		second = xRef*numpy.exp(xRef)/numpy.expm1(xRef) 
		f = self.ReturnMBB1(nu, mbeta_d1, mT_d1, betas_d1, Ts_d1) * (first - second)/(mT_d1 + Ts_d1)
		return f

	def ReturnPLSynch(self, nu, m_beta_s, betas_s):
		beta_s = (m_beta_s + betas_s )*self.sigma_beta_s
		gnu = self.Return_gnu(nu)	
		f = pow( nu / self.nuRef_synch, beta_s ) *gnu
		return f

	def ReturnPLSynch_DiffBeta(self, nu, m_beta_s, betas_s):
		f = numpy.log(nu / self.nuRef_synch) * self.ReturnPLSynch(nu, m_beta_s, betas_s)*self.sigma_beta_s
		return f
