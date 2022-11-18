import numpy
from astropy.modeling.blackbody import blackbody_nu as B
from astropy import units
import scipy.constants.constants as constants

class MBB1App01Beta():
	def __init__(self):
		self.name = 'dust_MBB1_app01_beta'
		self.nuRef_dust = 353.0
		self.sigma_T_d1 = 2.254
		self.sigma_beta_d1 = 0.0542
		self.T_CMB = 2.725
		self.n_params = 1
		self.means =numpy.array([27.77])
		self.param_names = ['beta_d1']

	def Return_gnu(self, nu):
		x_CMB =constants.h*nu*1.0e9/constants.k/self.T_CMB 
		gnu = pow(numpy.expm1(x_CMB) ,2) / ( pow(x_CMB,2) * numpy.exp(x_CMB))
		return gnu

	def ReturnMBB1(self, nu, means , params):
		mbeta_d1 = means[0]
		betas_d1 = params[0]
		beta_d = (mbeta_d1 ) * self.sigma_beta_d1
		mT_d1 =  8.873
		T_d = (mT_d1 ) * self.sigma_T_d1

		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h * self.nuRef_dust * 1.0e9/constants.k / T_d
		gnu = self.Return_gnu(nu)	
		f = numpy.ones(len(betas_d1))
		f += numpy.log( nu  / self.nuRef_dust ) * mbeta_d1  *self.sigma_beta_d1
		f *= gnu * pow( nu / self.nuRef_dust , beta_d +1) * numpy.expm1(xRef)/numpy.expm1(x)  	
		return f

	def ReturnFuncList(self):
		return [self.ReturnMBB1]
class MBB1App01():
	def __init__(self):
		self.name = 'dust_MBB1_app01'
		self.nuRef_dust = 353.0
		self.sigma_T_d1 = 2.254
		self.sigma_beta_d1 = 0.0542
		self.T_CMB = 2.725
		self.n_params = 2
		self.means =numpy.array([27.77, 8.873])
		self.param_names = ['beta_d1', 'T_d1']

	def Return_gnu(self, nu):
		x_CMB =constants.h*nu*1.0e9/constants.k/self.T_CMB 
		gnu = pow(numpy.expm1(x_CMB) ,2) / ( pow(x_CMB,2) * numpy.exp(x_CMB))
		return gnu

	def ReturnMBB1(self, nu, means , params):
		mbeta_d1 = means[0]
		mT_d1 = means[1]
		betas_d1 = params[0]
		Ts_d1 = params[1]
		beta_d = (mbeta_d1 ) * self.sigma_beta_d1
		T_d = (mT_d1 ) * self.sigma_T_d1

		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h * self.nuRef_dust * 1.0e9/constants.k / T_d
		gnu = self.Return_gnu(nu)	
		f = numpy.ones(len(betas_d1))
		f += numpy.log( nu  / self.nuRef_dust ) * mbeta_d1  *self.sigma_beta_d1
		first = x*numpy.exp(x)/numpy.expm1(x)
		second = xRef*numpy.exp(xRef)/numpy.expm1(xRef) 
		f += ( first - second )/(mT_d1) * Ts_d1

		f *= gnu * pow( nu / self.nuRef_dust , beta_d +1) * numpy.expm1(xRef)/numpy.expm1(x)  	
		return f

	def ReturnFuncList(self):
		return [self.ReturnMBB1]

class MBB1Beta():
	def __init__(self):
		self.name = 'dust_MBB1'
		self.nuRef_dust = 353.0
		self.sigma_T_d1 = 2.254
		self.sigma_beta_d1 = 0.0542
		self.T_CMB = 2.725
		self.n_params = 1
		self.means =numpy.array([27.77])
		self.param_names = ['beta_d1']
		self.T_d = 8.873


	def Return_gnu(self, nu):
		x_CMB =constants.h*nu*1.0e9/constants.k/self.T_CMB 
		gnu = pow(numpy.expm1(x_CMB) ,2) / ( pow(x_CMB,2) * numpy.exp(x_CMB))
		return gnu

	def ReturnMBB1(self, nu, means , params):
		mbeta_d1 = means[0]
		betas_d1 = params[0]
		beta_d = (mbeta_d1 + betas_d1) * self.sigma_beta_d1
		T_d = ( self.T_d ) * self.sigma_T_d1
		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h * self.nuRef_dust * 1.0e9/constants.k / T_d
		gnu = self.Return_gnu(nu)	
		f = gnu * pow( nu / self.nuRef_dust , beta_d +1) * numpy.expm1(xRef)/numpy.expm1(x)
		return f


	def ReturnMBB1_DiffBeta(self, nu, means , params):
		mbeta_d1 = means[0]
		betas_d1 = params[0]
		f = self.ReturnMBB1( nu, means , params) * numpy.log( nu  / self.nuRef_dust )*self.sigma_beta_d1
		return f

	def ReturnFuncList(self):
		return [self.ReturnMBB1, self.ReturnMBB1_DiffBeta]

class MBB1():
	def __init__(self):
		self.name = 'dust_MBB1'
		self.nuRef_dust = 353.0
		self.sigma_T_d1 = 2.254
		self.sigma_beta_d1 = 0.0542
		self.T_CMB = 2.725
		self.n_params = 2
		self.means =numpy.array([27.77, 8.873])
		self.param_names = ['beta_d1', 'T_d1']


	def Return_gnu(self, nu):
		x_CMB =constants.h*nu*1.0e9/constants.k/self.T_CMB 
		gnu = pow(numpy.expm1(x_CMB) ,2) / ( pow(x_CMB,2) * numpy.exp(x_CMB))
		return gnu

	def ReturnMBB1(self, nu, means , params):
		mbeta_d1 = means[0]
		mT_d1 = means[1]
		betas_d1 = params[0]
		Ts_d1 = params[1]
		beta_d = (mbeta_d1 + betas_d1) * self.sigma_beta_d1
		T_d = (mT_d1 + Ts_d1) * self.sigma_T_d1
		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h * self.nuRef_dust * 1.0e9/constants.k / T_d
		gnu = self.Return_gnu(nu)	
		f = gnu * pow( nu / self.nuRef_dust , beta_d +1) * numpy.expm1(xRef)/numpy.expm1(x)
		return f

	def ReturnMBB1_DiffT(self, nu, means , params):
		mbeta_d1 = means[0]
		mT_d1 = means[1]
		betas_d1 = params[0]
		Ts_d1 = params[1]
		beta_d = (mbeta_d1 + betas_d1) * self.sigma_beta_d1
		T_d = (mT_d1 + Ts_d1) * self.sigma_T_d1
		x = constants.h*nu*1.0e9/constants.k/T_d
		xRef = constants.h * self.nuRef_dust * 1.0e9/constants.k / T_d
		first= x*numpy.exp(x)/numpy.expm1(x)
		second = xRef*numpy.exp(xRef)/numpy.expm1(xRef) 
		f = self.ReturnMBB1( nu, means , params) * (first - second)/(mT_d1 + Ts_d1)
		return f

	def ReturnMBB1_DiffBeta(self, nu, means , params):
		mbeta_d1 = means[0]
		mT_d1 = means[1]
		betas_d1 = params[0]
		Ts_d1 = params[1]
		f = self.ReturnMBB1( nu, means , params) * numpy.log( nu  / self.nuRef_dust )*self.sigma_beta_d1
		return f
	def ReturnFuncList(self):
		return [self.ReturnMBB1, self.ReturnMBB1_DiffBeta,self.ReturnMBB1_DiffT]

class PLSynchApp01():
	def __init__(self):
		self.name = 'synch_PL'
		self.nuRef_synch = 30.0
		self.sigma_beta_s = 0.0632
		self.T_CMB = 2.725
		self.n_params = 1
		self.means = numpy.array([ -47.47])
		self.param_names = ['beta_s']

	def Return_gnu(self, nu):
		x_CMB =constants.h*nu*1.0e9/constants.k/self.T_CMB 
		gnu = pow(numpy.expm1(x_CMB),2) / ( pow(x_CMB,2) * numpy.exp(x_CMB))
		return gnu

	def ReturnPLSynch(self, nu, means, params):
		m_beta_s = means[0]
		betas_s = params[0]
		beta_s = (m_beta_s + betas_s )*self.sigma_beta_s
		gnu = self.Return_gnu(nu)	
		f = numpy.ones( len(betas_s) )
		f += numpy.log(nu / self.nuRef_synch) * betas_s * self.sigma_beta_s
		f *=  pow( nu / self.nuRef_synch, beta_s ) *gnu
		return f

	def ReturnFuncList(self):
		return [self.ReturnPLSynch]

class PLSynch():
	def __init__(self):
		self.name = 'synch_PL'
		self.nuRef_synch = 30.0
		self.sigma_beta_s = 0.0632
		self.T_CMB = 2.725
		self.n_params = 1
		self.means = numpy.array([ -47.47])
		self.param_names = ['beta_s']

	def Return_gnu(self, nu):
		x_CMB =constants.h*nu*1.0e9/constants.k/self.T_CMB 
		gnu = pow(numpy.expm1(x_CMB),2) / ( pow(x_CMB,2) * numpy.exp(x_CMB))
		return gnu

	def ReturnPLSynch(self, nu, means, params):
		m_beta_s = means[0]
		betas_s = params[0]
		beta_s = (m_beta_s + betas_s )*self.sigma_beta_s
		gnu = self.Return_gnu(nu)	
		f = pow( nu / self.nuRef_synch, beta_s ) *gnu
		return f

	def ReturnPLSynch_DiffBeta(self, nu, means, params):
		m_beta_s = means[0]
		betas_s = params[0]
		f = numpy.log(nu / self.nuRef_synch) * self.ReturnPLSynch(nu, means, params)*self.sigma_beta_s
		return f
	def ReturnFuncList(self):
		return [self.ReturnPLSynch, self.ReturnPLSynch_DiffBeta]
