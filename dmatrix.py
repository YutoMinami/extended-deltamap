import sympy
import numpy

class DMatrix :
	def __init__(self, verbose = False):    
		self.dim_params = 0
		self.d_params = []
		self.d_funcs  =[]
		self.freqs = None
		self.width = None
		self.n_freqs = None
		self.D_matrix_template=[]
		self.D_matrix = None
		self.gnu = None
		self.verbose = verbose
		pass
	def SetGnu(self, gnu):
		self.gnu = gnu

	def AddD(self, f_D):
		#count params
		symbols = f_D.free_symbols
		params =[]
		self.dim_params+=1
		for param in symbols:
			if 'nu' not in param.name:
				self.dim_params +=1
				params.append(param)
		self.d_params.append(params)
		self.d_funcs.append(f_D)

	def SetFreqs(self,freqs, width):
		self.freqs = numpy.asarray(freqs)
		self.width = numpy.asarray(width)
		self.n_freqs = len(freqs)

	def DiffD(self):
		#differentiate D
		for func, params in zip(self.d_funcs, self.d_params):
			self.D_matrix_template.append(func)
			for param in params:
				self.D_matrix_template.append(
					sympy.simplify(
							sympy.diff(func, param)
						)
				)
	def PrepareOneDiffDMatrix(self, diff_param):
		for func, params in zip(self.d_funcs, self.d_params):
			self.D_matrix_template.append(func)
			for param in params:
				if diff_param in param.name:
					self.D_matrix_template.append(
						sympy.simplify(
								sympy.diff(func, param)
							)
					)
				else:
					continue
		l_Dmatrix =[]
		for nu in self.freqs:
			nu_D = [ func.subs('nu', nu) for func in self.D_matrix_template]
			l_Dmatrix.append(nu_D)
		if self.verbose:
			print(self.dim_params, self.n_freqs)
			print((l_Dmatrix))
		self.D_matrix =sympy.Matrix(l_Dmatrix)	
	def PrepareUniformDMatrix(self):
		for func, params in zip(self.d_funcs, self.d_params):
			self.D_matrix_template.append(func)
		l_Dmatrix =[]
		for nu in self.freqs:
			nu_D = [ func.subs('nu', nu) for func in self.D_matrix_template]
			l_Dmatrix.append(nu_D)
		if self.verbose:
			print(self.dim_params, self.n_freqs)
			print((l_Dmatrix))
		self.D_matrix =sympy.Matrix(l_Dmatrix)	

	def PrepareDMatrix(self):
		self.DiffD()
		l_Dmatrix =[]
		for nu in self.freqs:
			nu_D = [ func.subs('nu', nu)  for func in self.D_matrix_template]
			l_Dmatrix.append(nu_D)
		if self.verbose:
			print(self.dim_params, self.n_freqs)
			print((l_Dmatrix))
		#self.D_matrix =sympy.Matrix(self.n_freqs, self.dim_params, l_Dmatrix)	
		self.D_matrix =sympy.Matrix(l_Dmatrix)	
	def CalcDMatrix(self):
		#calculate n_freqs of Ds
		#calculate n_params of Ds
		pass
