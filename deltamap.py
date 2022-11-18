import sympy
import dmatrix
import scipy
import numpy
from iminuit import Minuit
import time
from scipy.linalg import lapack
class DeltaMap:
	"""
	
	"""
	def __init__(self, verbose = False):
		"""
		Parameters 
		---------------------
		hoge()
		
		Returns
		----------------------
		huga()
		"""
		self.verbose = verbose
		self.S0_CMB_SM = None
		self.S0_CMB_BSM = None
		self.S0_CMB = None
		self.S0_CMB_I = None
		self.S0_CMB_L  =None
		#self.S0_CMB_LU = None
		self.size = None
		self.params = None
		self.dmtrx = None
		self.mvec = None
		self.meanvec = None
		#self.tmpmvec = None
		#self.Am_vec = None
		self.N_array = None
		self.N_list = None
		self.NI_list = None
		#self.Nmatrix_I_array = None
		self.N_inv_array = None
		self.inits = None
		#self.limits = None
		self.valid = True

		self.param_values=None
		self.param_errors = None
		self.fg_valid = False
		self.r_valid = False
		self.r_minos = None
		self.A = None
		self.AL= None
		self.ALU = None
		self.AP = None
		self.H =None
		#self.HL = None
		#self.Y = None
		#self.Middle = None
		self.B = None
		self.BLU = None
		self.BP = None
		#self.BpD = None
		#self.BpDLU = None
		#self.BpDP = None
		self.BI = None
		self.D =None
		self.DiffDs = None
		self.Delta = None
		self.DNIDL = None
		self.DT_SpNI_DL = None
		#self.NI =None
		self.NIm =None
		self.Hm = None
		#self.HNm = None
		#self.HN = None
		#self.S0INI_I = None
		self.AI_NIm = None
		self.tmp_likelihood = 1.0e3
		self.withxRPrior = False
		self.meanxR = 0.81 #K
		self.sigmaxR = 0.15 #K
		self.withTdPrior = False
		self.meanTd = 20.9 #K
		self.sigmaTd = 5.0 #K
		self.is_noise_matrix =  False
		self.is_noise_set = False
		self.inds_cache = {}
		self.migrad = True
		self.r_verbose = False
		    

	def SetxRPrior(self, meanxR, sigmaxR):
		self.meanxR = meanxR
		self.sigmaxR = sigmaxR
	def SetTdPrior(self, meanTd, sigmaTd):
		self.meanTd = meanTd
		self.sigmaTd = sigmaTd

	def SetS0(self, S0_SM,S0_BSM):
		self.S0_CMB_SM = numpy.copy(S0_SM)
		self.S0_CMB_BSM = numpy.copy(S0_BSM)
		self.size = self.S0_CMB_SM.shape[0]
		pass

	def SetFgDmatrix(self, dmtrx):
		self.dmtrx = dmtrx
	
	def DiffDmatrix(self):
		self.DiffDs = []
		for param in self.params:
			if 'r' in param.name:
				continue
			else:
				self.DiffDs.append(
					sympy.simplify(
							sympy.diff(self.dmtrx.D_matrix, param)
						)
				)

	def CheckFitParams(self):
		if self.S0_CMB_SM is None  or self.S0_CMB_BSM is None or self.dmtrx is None:
			raise Exception('Have to do SetSO() and SetFgDmatrix() before!')
		self.params = []
		#for param in self.S0_CMB.free_symbols:
		r = sympy.Symbol('r')
		self.params.append( r )
		for param in self.dmtrx.D_matrix.free_symbols:
			self.params.append(param)
		pass

	def SetMvec(self, mvec):
		tmp_mvec = []
		for mvec_i in  mvec:
			tmp_mvec.append(mvec_i.reshape(len(mvec_i), 1) )
		self.mvec = tmp_mvec

	def SetNoiseArray(self,narr):
		self.N_array = numpy.copy(narr)
		self.is_noise_set = True
		self.is_noise_matrix = False

	def SetNoiseList(self, N_list):
		"List of Noise Cov Matrix"
		self.N_list = N_list
		self.is_noise_set = True
		self.is_noise_matrix = True
		
	def CheckNoiseDim(self):
		if not self.is_noise_set:
			raise Exception('Noise array or list of matrix is not set!')
		nfreq_dmatrix = self.dmtrx.D_matrix.shape[0]
		if self.is_noise_matrix:
			nfreq_noise = len(self.N_list)
		else:
			nfreq_noise = self.N_array.shape[0] 
		if nfreq_noise != nfreq_dmatrix:
			raise Exception('Number of frequencies from noise and dmatrix are different!')
		pass
	
	def CalcNInvArray(self):
		if self.is_noise_matrix:
			self.NI_list =  []
			for idx,matrix in enumerate(self.N_list):
				#self.NI_list.append( scipy.linalg.inv(matrix) )
				self.NI_list.append( self.fast_positive_definite_inverse(matrix) )
		else:
			self.N_inv_array = 1./self.N_array
			self.NI_list = []
			for i in self.N_inv_array:
				#self.NI_list.append(i * sympy.eye(self.size) )
				self.NI_list.append(i * numpy.identity(self.size) )

	#def SetParameterInitial(self, names, inits, limits=None):
	def SetParameterInitial(self, inits):
		if len(self.params) != len(inits):
			raise Exception('The size of params and inits should be the same.')
		"""
		if limits is None:
			limits = [] 
			for init in inits:
				limits.append(  (None,None )  )
		"""
		self.inits = inits
		self.param_values = {}
		#self.limits=[]
		for key in inits.keys():
			self.param_values[key] = inits[key][0]
		"""
		for name,init,limit in zip(names, inits, limits):
			#self.inits.append(init)
			self.inits[name] = init
			self.param_values[name] = init
			self.limits.append(limit)
		"""
		if self.verbose :
			print(self.param_values)
			print(self.inits)
			#print(self.limits)
		pass

	def SetParameters(self, values):
		for key in values.keys():
			if key in self.param_values:
				self.param_values[key] = values[key]
		pass

	def CalcNIm(self):
		self.NIm = None
		if self.is_noise_matrix:
			for ni_i,mvec_i in zip(self.NI_list, self.mvec):
				if self.NIm is None:
					self.NIm = ni_i@mvec_i
				else:
					self.NIm += ni_i@mvec_i
			if self.verbose:
				print('NIm: '+ str(self.NIm))
		else:
			for nval,mvec_i in zip(self.N_inv_array, self.mvec):
				if self.NIm is None:
					self.NIm =  nval* mvec_i
				else:
					self.NIm += nval*mvec_i
			if self.verbose:
				print('NIm: '+ str(self.NIm))
	
	def Calcni(self):
		self.ni = None
		for i in self.NI_list:
			if self.ni is None:
				self.ni = i
			else:
				self.ni += i
	
	"""
	def CalcNIM(self):
		self.NIM = None
		for nval,mvec_i in zip(self.N_inv_array, self.meanvec):
			if self.NIM is None:
				self.NIM =  nval* mvec_i
			else:
				self.NIM += nval*mvec_i
		#self.NIM = self.NIM.reshape(len(self.NIM),1) 
		if self.verbose:
			print('NIM: '+ str(self.NIM))
	"""
	def ReturnMXCtXM(self):
		second = self.HM + self.Delta * scipy.linalg.cho_solve(  (self.BL,True) ,self.HM)
		second = second - self.ni @  scipy.linalg.cho_solve(  (self.AL,True) ,self.HM)
		XM = -self.NIM + second
		return (XM.T @ self.S0_CMB_BSM @ XM)[0,0]

	def ReturnTrace(self):
		tmp = self.ni @ self.S0_CMB_BSM 
		return numpy.trace( tmp -  self.ni @ scipy.linalg.cho_solve(  (self.AL,True) , tmp)     )
	"""
	def CalcBI_HM(self):
		D = self.dmtrx.D_matrix
		for param in self.params:
			#self.D = self.D.subs(param, value)
			D = D.subs(param, self.param_values[param.name])
			if self.verbose:
				print(str(param)+': ', value)
		D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 
		NI = numpy.diag(self.N_inv_array)  

		self.BI_HM = scipy.linalg.cho_solve(  (self.BL,True) ,self.HM)
	"""
	"""
	def ReturnDpYM(self, Dp):
		D = self.dmtrx.D_matrix
		Dpsub = Dp
		for param in self.params:
			#self.D = self.D.subs(param, value)
			D = D.subs(param, self.param_values[param.name])
			Dpsub = Dpsub.subs(param, self.param_values[param.name])

			if self.verbose:
				print(str(param)+': ', value)
		D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 
		Dpsub = sympy.matrix2numpy(Dpsub, dtype = numpy.float64 ) 
		NI = numpy.diag(self.N_inv_array)  

		left_tmp = Dpsub@( scipy.linalg.cho_solve((self.DNIDL, True) , D.T @ NI) )
		vec= [ ni@(mvec_i - self.BI_HM) for mvec_i,ni in zip(self.meanvec, self.NI_list )]

		vec_left = []
		for  idx in range(left_tmp.shape[0]):
			vec_left.append( sum(left_tmp[idx,:] * vec ) )

		vec_right = [ mvec_i - self.BI_HM for mvec_i,ni in zip(self.meanvec, self.NI_list )]
		right_tmp = []
		for idx in range(self.H.shape[0]):
			right_tmp.append( sum( self.H[idx,:] * vec_right ) )
		vec_right = right_tmp

		right_tmp = []
		for mvec_i in vec_right:
			right_tmp.append( mvec_i  -  scipy.linalg.cho_solve((self.BL, True), sum(vec_right)  ) )
		vec_right = right_tmp
		vec_right = [ vec_i - ni@mvec_i   for vec_i,mvec_i,ni in zip(vec_right, self.meanvec, self.NI_list)]
		
		scalar = 0.
		for left,right in zip(vec_left, vec_right ):
			scalar += (left.T @ right)[0,0]
		return scalar
	"""


	"""
	def CalcAm(self):
		M = None
		for ninv, mvec_i in zip(self.N_inv_array, self.mvec):
			if M is None:
				M = ninv*mvec_i
			else:
				M += ninv*mvec_i
		AM =  scipy.linalg.lu_solve((self.ALU, self.AP), M)
		self.Am_vec =[]
		for ninv, mvec_i in zip(self.N_inv_array,self.mvec):
			self.Am_vec.append( (mvec_i - AM)*ninv )

	"""

	"""
	def CalcMiddle(self):
		first = numpy.kron( numpy.ones( (self.size, self.size) ),self.Y)
		self.BI = scipy.linalg.lu_solve( (self.BLU, self.BP ), numpy.eye(self.B.shape[0]) ) 
		nvec = self.N_inv_array.reshape(1, len(self.N_inv_array) )
		ivecY = nvec @self.Y
		YIIY = ivecY.T @ ivecY
		second = numpy.kron( YIIY  ,self.BI )
		self.Middle = first - second
	"""
	"""
	def Return_mNIAm(self):
		mNIAm = None
		for left,right in zip(self.mvec, self.Am_vec):
			if mNIAm is None:
				mNIAm = ( left.T@right)
			else:
				mNIAm += ( left.T@right)
		return mNIAm[0,0]
	"""
	"""
	def Return_mAMiddleAm(self):
		Am = numpy.vstack(self.Am_vec)
		return -Am.T@(self.Middle@Am)[0,0]
	"""
	def initialise(self):
		self.CheckFitParams()
		self.CheckNoiseDim()
		self.CalcNInvArray()
		self.CalcNIm()
		pass


	"""
	From here, functions for each iteration
	self.param_values are updated in evenry iteration
	"""

	def CalcCMB0Inverse(self):
		"""
		r = sympy.Symbol('r')
		#for param,value in zip(self.params, self.param_values):
		for param in self.params:
			 r = r.subs(param, self.param_values[param.name])
		if self.verbose:
			print('r is :'+str(r) )
		r = float(r.evalf())	
		"""
		r = self.param_values['r']
		self.S0_CMB = self.S0_CMB_SM + r* self.S0_CMB_BSM
			
		precho = time.time()
		self.valid = self.is_pos_def(self.S0_CMB)
		if not self.valid :
			print('S0_CMB is not positive definite')
			print(self.param_values['r'])
			return

		"""
		self.S0_CMB_LU,_ = scipy.linalg.lu_factor(self.S0_CMB)
		self.S0_CMB_I = scipy.linalg.lu_solve( (self.S0_CMB_LU,_) , numpy.identity( self.S0_CMB.shape[0] ))
		"""
		#S0_CMB_LU,_ = scipy.linalg.lu_factor(self.S0_CMB)
		#self.S0_CMB_I = scipy.linalg.lu_solve( (S0_CMB_LU,_) , numpy.identity( self.S0_CMB.shape[0] ))
		self.S0_CMB_L = scipy.linalg.cholesky( self.S0_CMB , True)
		#self.S0_CMB_I = scipy.linalg.cho_solve( (self.S0_CMB_L,True) , numpy.identity( self.S0_CMB.shape[0] ))
		self.S0_CMB_I = self.fast_positive_definite_inverse( self.S0_CMB )

		if self.verbose:
			print(self.S0_CMB_I)
		postcho = time.time()
		if numpy.linalg.cond( self.S0_CMB ) > 1./numpy.finfo( self.S0_CMB.dtype).eps:
			print('S0_CMB is invalid')
			print(numpy.linalg.cond( self.S0_CMB ))
			self.valid = False

		if self.verbose:
			"""
			prelu = time.time()
			tmp_S0_CMB_I = CMB_tmp.inv( 'LU')
			postlu = time.time()
			"""
			print('With scipy ', str(postcho - precho), ' s' )
			#print('With LU ', str(postlu - prelu), ' s' )
			#print('inv equals ? : ', tmp_S0_CMB_I.equals( self.S0_CMB_I ) )

	def CalcA(self):
		self.A = None
		#self.A = self.S0_CMB_I
		##if numpy array case
		self.A = numpy.copy(self.S0_CMB_I)
		"""
		if not numpy.array_equal(self.A, self.A.T):
			self.A = (self.A+self.A.T)/2.
		"""
		for i in self.NI_list:
			self.A += i
		if self.verbose:
			print('A shape :' + str(self.A.shape))
			print('A :' + str(self.A))
		self.AL  = scipy.linalg.cholesky(self.A , True)
		#self.ALU, self.AP  = scipy.linalg.lu_factor(self.A)
		"""
		if not self.is_pos_def(self.A):
			self.valid = False
			return
		"""
		if numpy.linalg.cond( self.A ) > 1./numpy.finfo( self.A.dtype).eps:
			print('A is invalid')
			self.valid = False
	
	def CalcH_matrix(self):
		D = self.dmtrx.D_matrix
		for param in self.params:
			#self.D = self.D.subs(param, value)
			D = D.subs(param, self.param_values[param.name])
			if self.verbose:
				print(str(param)+': ', self.param_values[param.name])
		D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 
		matrix_list = []
		for i in range( self.dmtrx.D_matrix.shape[1]):
			i_list =[]
			for j in range( self.dmtrx.D_matrix.shape[1]):
				element = numpy.zeros_like(self.NI_list[0])
				for k in range( len( self.NI_list)):
					element += D[k,i] * self.NI_list[k] * D[k,j]
				i_list.append(element)

			matrix_list.append(i_list)

		DTNID = numpy.block( matrix_list)
		self.DNIDL = scipy.linalg.cholesky( DTNID , lower =True)


		#self.H = (NI@D) @ ( scipy.linalg.cho_solve((self.DNIDL, True) , D.T @ NI) ) 

	#def CalcDTNIDc(self):
		matrix_list = []
		for i in range( self.dmtrx.D_matrix.shape[1]):
			i_list =[]
			element = numpy.zeros_like(self.NI_list[0])
			for j in range( len( self.NI_list)):
				element += D[j,i] * self.NI_list[j]
			i_list.append(element)
			matrix_list.append(i_list)
		self.DTNIDc = numpy.block(matrix_list)

		DT_SpNI_D = DTNID - self.DTNIDc @ scipy.linalg.cho_solve( (self.AL, True), self.DTNIDc.T)
		self.DT_SpNI_DL = scipy.linalg.cholesky(DT_SpNI_D, lower= True)


	#def CalcDTNIM(self):
		matrix_list = []
		for i in range( self.dmtrx.D_matrix.shape[1]):
			i_list =[]
			
			#element = numpy.zeros_like(self.NI_list[0])
			element = numpy.zeros_like(  self.meanvec[0] )
			for j in range( len( self.NI_list)):
				element += D[j,i] * (self.NI_list[j]@ self.meanvec[j])
			i_list.append(element)
			matrix_list.append(i_list)
		self.DTNIM = numpy.block(matrix_list)

	def CalcDTNID_I_DTNIM(self):
		self.DTNID_I_DTNIM = scipy.linalg.cho_solve(  (self.DNIDL,True) , self.DTNIM)

	def CalcDcTNID_DTNID_I_DTNIM(self):
		self.DcTNID_DTNID_I_DTNIM  = self.DTNIDc.T @ self.DTNID_I_DTNIM
	def CalcBIDcTNID_DTNID_I_DTNIM(self):
		self.BIDcTNID_DTNID_I_DTNIM  = scipy.linalg.cho_solve(  (self.BL,True) , self.DcTNID_DTNID_I_DTNIM)

	def CalcDelta(self):
		self.Delta = self.DTNIDc.T@scipy.linalg.cho_solve(  (self.DNIDL,True) , self.DTNIDc)
	def CalcH(self):
		"""
		self.D =None
		self.D = self.dmtrx.D_matrix
		"""

		D = self.dmtrx.D_matrix

		#for param,value in zip(self.params, self.param_values):
		for param in self.params:
			#self.D = self.D.subs(param, value)
			D = D.subs(param, self.param_values[param.name])
			if self.verbose:
				print(str(param)+': ', self.param_values[param.name])

		#self.D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 

		D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 


		if self.verbose:
			print('dtype of N_inv_array')
			print(self.N_inv_array.dtype)


		#self.NI = None

		#NI = sympy.diag(*N_inv_array)
		NI = numpy.diag(self.N_inv_array)  
		#self.NI = numpy.diag(N_inv_array)  

		#N_I = sympy.Matrix(numpy.diag(N_inv_array) )

		"""
		if self.verbose:
			print('dtype of NI')
			print(NI.dtype)
		"""

		#tmp = self.D.T @ (NI @ self.D )
		tmp = D.T @ (NI @ D )
		

		"""
		if not numpy.array_equal( tmp.T, tmp):
			tmp = (tmp.T + tmp)/2.
			if self.verbose:
				print('NI =')
				#print(self.NI)
				print(NI)
				print('DNID is not symmetric')
				pass
			pass
		

		"""
		#Method 2: in sympy calculate H from NI D
		"""
		if tmp.T != tmp:
			tmp = (tmp.T+tmp)/2.
		"""
		#Method 1: in numpy calculate H from tmp, NI, D with cholesky
		"""
		tmp = sympy.matrix2numpy(tmp, dtype = numpy.float64 ) 
		NI = sympy.matrix2numpy(NI, dtype = numpy.float64 ) 
		D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 
		"""

		if numpy.linalg.cond(tmp) > (1./numpy.finfo( tmp.dtype).eps ):
			print('DNID is invalid')
			self.valid = False
		#self.HL = None
		HL = None
		#self.HL = scipy.linalg.cholesky( tmp , lower =True)
		#self.HL = scipy.linalg.cholesky( tmp , lower =True)
		self.DNIDL = scipy.linalg.cholesky( tmp , lower =True)
		#self.DNIDLU,self.DNIDP = scipy.linalg.lu_factor( tmp )
		self.H  = None

		#self.H = (self.NI@self.D) @ ( scipy.linalg.cho_solve((self.HL, True) , self.D.T @ self.NI) ) 
		#self.H = (NI@D) @ ( scipy.linalg.cho_solve((self.HL, True) , D.T @ NI) ) 
				#self.H = (NI@D) @ ( scipy.linalg.solve(  tmp , D.T @ NI, sym_pos=True ) ) 
		#self.H = (NI@D) @ ( scipy.linalg.cho_solve((self.HL, True) , tmp @ D.T @ NI) ) 
		#self.H = (NI@D) @ ( scipy.linalg.cho_solve((self.HL, True) , tmp.T @ D.T @ NI) ) 
		#self.H = ( scipy.linalg.cho_solve((self.HL, True) , self.D.T @ self.NI) )
		self.H = (NI@D) @ ( scipy.linalg.cho_solve((self.DNIDL, True) , D.T @ NI) ) 
		#self.H = (NI@D) @ ( scipy.linalg.lu_solve((self.DNIDLU, self.DNIDP) , D.T @ NI) ) 
		"""
		"""


		"""
		#self.H = NI@D@( (tmp.T@tmp).cholesky_solve(tmp.T@D.T@NI) )
		self.H = NI@D@( (tmp).cholesky_solve( D.T@NI ) )
		#self.H = sympy.cancel(self.H)
		"""

		"""
		for param in self.params:
		#for param,value in zip(self.params, self.param_values):
			self.H = self.H.subs(param, self.param_values[param.name])
			if self.verbose:
				print(param.name +  ': ', self.param_values[param.name])
	
		self.H = sympy.matrix2numpy(self.H, dtype = numpy.float64) 
		"""
		
		#TODO this is not needed !
		"""		
		if not numpy.array_equal( self.H.T, self.H):
			self.H = ( self.H.T + self.H)/2.
		"""

		if numpy.isnan(self.H).any() :
			print('H contains invalid value')
			self.valid = False

		if self.verbose:
			print('H matrix: ' +str(self.H))
		self.Delta = self.H.sum(dtype = numpy.float64)

		"""
		tmpH = numpy.copy(self.H)
		tmpH = tmpH.astype('float64') 
		self.Delta = tmpH.sum()
		"""

		if self.verbose:
			print('Delta : ' +str(self.Delta))
		pass

	def CalcHm(self):	
		self.Hm = None

		## with numpy array
		#tmpH = numpy.copy(self.H)

		for H_i, mvec_i in zip( self.H.sum(axis=0),self.mvec):
			if self.Hm is  None:
				#self.Hm =  H_i*numpy.eye(len(mvec_i)) @ mvec_i
				self.Hm =  H_i*mvec_i
			else:
				#self.Hm += H_i*numpy.eye(len(mvec_i)) @ mvec_i
				self.Hm += H_i * mvec_i
		#self.Hm = sympy.Matrix(self.Hm)
		if self.verbose:
			print('Hm: '+ str(self.Hm))
		pass

	def CalcHM(self):	
		self.HM = None

		for H_i, mvec_i in zip( self.H.sum(axis=0),self.meanvec):
			if self.HM is  None:
				#self.HM =  H_i*numpy.eye(len(mvec_i)) @ mvec_i
				self.HM =  H_i*mvec_i
			else:
				#self.HM += H_i*numpy.eye(len(mvec_i)) @ mvec_i
				self.HM += H_i * mvec_i
		#self.HM = sympy.Matrix(self.HM)
		if self.verbose:
			print('HM: '+ str(self.HM))

	def CalcB(self):
		self.B = None
		self.BLU = None
		self.BP = None
		self.BpD = None
		self.BpDLU = None
		self.BpDP = None
		"""
		"""

		#self.B = self.Delta*sympy.eye(self.size) - self.A
		if self.is_noise_matrix:
			self.B = self.Delta - self.A
		else:
			self.B = self.Delta* numpy.identity(self.size) - self.A
			self.BpD = numpy.identity(self.size) - self.A/self.Delta 
		#self.BL = scipy.linalg.cholesky(self.B)
		#self.BLU, self.BP = scipy.linalg.lu_factor(self.B.T @ self.B)
		self.BLU, self.BP = scipy.linalg.lu_factor(self.B)
		self.BL = scipy.linalg.cholesky(-self.B , True)
		if not self.is_noise_matrix:
			self.BpDL = scipy.linalg.cholesky(-self.BpD,True)
			##TODO###
			self.BpDLU, self.BpDP = scipy.linalg.lu_factor(self.BpD)
			#self.BpDLU, self.BpDP = scipy.linalg.lu_factor(self.BpD.T @ self.BpD)
			if numpy.linalg.cond( self.BpD ) > 1./numpy.finfo( self.BpD.dtype).eps:
				print('BpD is invalid')
				self.valid = False

		if numpy.linalg.cond( self.B ) > 1./numpy.finfo( self.B.dtype).eps:
			print('B is invalid')
			self.valid = False
		"""
		"""
		if self.verbose:
			print('B shape :' + str(self.B.shape))
			print('B :')
			print(self.B)
		pass

	def CalcAI_NIm(self):
		self.AI_NIm = None
		if self.verbose:
			pretime = time.time()
		"""
		for nval,mvec_i in zip(self.N_array, self.mvec):
		#for nval,mvec_i in zip(self.N_inv_array, self.mvec):
			tmpA =  self.A*nval
			tmpLU,tmpAP = scipy.linalg.lu_factor(tmpA)
			if self.AI_NIm is None:
				self.AI_NIm = scipy.linalg.lu_solve( (tmpLU, tmpAP),  mvec_i  )
			else:
				self.AI_NIm += scipy.linalg.lu_solve( (tmpLU, tmpAP), mvec_i  )

		"""

		self.AI_NIm = scipy.linalg.cho_solve((self.AL, True), self.NIm)
		#self.AI_NIm = scipy.linalg.lu_solve((self.ALU, self.AP), self.NIm)
		"""
		for nval,mvec_i in zip(self.N_array, self.mvec):
			tmpA =  (nval * self.A).T@(nval * self.A)
			tmpLU,tmpAP = scipy.linalg.lu_factor(tmpA)
			if self.AI_NIm is None:
				self.AI_NIm = scipy.linalg.lu_solve( (tmpLU, tmpAP), (nval*self.A)@mvec_i  )
			else:
				self.AI_NIm += scipy.linalg.lu_solve( (tmpLU, tmpAP), (nval*self.A)@mvec_i  )
		"""
		if self.verbose:
			posttime= time.time()
		if self.verbose:
			print(self.AI_NIm)
			print('processing time of AI_NIm ', (posttime- pretime) )
		"""	
		self.meanNI = self.N_inv_array.mean() 
		self.ApLU, self.ApP = scipy.linalg.lu_factor(self.A/self.meanNI)
		self.AI_NIm = scipy.linalg.lu_solve((self.ApLU, self.ApP), self.NIm/self.meanNI)
		"""

	def OrthogonalComplement(self, x, threshold=1e-15):
		r, c = x.shape
		if c < r:
			import warnings
			warnings.warn('fewer rows than columns', UserWarning)
		s, v, d = scipy.linalg.svd(x)
		rank = (v > threshold).sum()
		oc = d[rank:,:]
		k_oc = oc.shape[0]
		#oc = oc.dot( scipy.linalg.inv(oc[:k_oc, :]))
		oc = ( scipy.linalg.inv(oc[:, :k_oc])).dot(oc)
		for i in range(oc.shape[0]):
			oc[i] = oc[i] / numpy.sqrt(oc[i].dot( oc[i] ) )
		return oc.T
	def CalcTmpVec(self):
		D = self.dmtrx.D_matrix
		for param in self.params:
			D = D.subs(param, self.param_values[param.name])
		D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 
		oc = self.OrthogonalComplement(D.T)
		oc_sum =  oc.sum(axis=1)
		self.tmpmvec  = [ oc_i * mvec_i  for mvec_i, oc_i in zip(self.mvec, oc_sum) ]

	def CalcMeanVec(self):
		self.meanvec = [ mvec_i - self.AI_NIm for mvec_i in self.mvec]
		#self.meanvec = [ mvec_i - self.AI_NIm for mvec_i in self.tmpmvec]

	def ReturnMHM(self):
		if self.is_noise_matrix:
			return -( self.DTNIM.T @ self.DTNID_I_DTNIM )[0,0]
		else:
			MHM = None
			for idx in range( self.H.shape[1] ):
				left_mvec = self.meanvec[idx]
				right = None
				for H_i, mvec_i in zip( self.H[idx,:], self.meanvec):
					if right is None:
						right = H_i *  mvec_i
					else:
						right += H_i * mvec_i
				if MHM is None:
					MHM = left_mvec.T @ right
				else:
					MHM += left_mvec.T @right

			return -MHM[0,0]
	def ReturnMHBIHM(self):
		if self.is_noise_matrix:
			return -( self.DcTNID_DTNID_I_DTNIM.T @ self.BIDcTNID_DTNID_I_DTNIM)[0,0]	
		else:
			HM =  None
			for H_i, mvec_i in zip( self.H.sum(axis=0),self.meanvec ):
				if HM is  None:
					HM =  H_i * mvec_i
				else:
					HM += H_i * mvec_i
			#return (HM.T @ scipy.linalg.lu_solve(  (self.BLU, self.BP) ,HM)  )[0,0]
			return - (HM.T @ scipy.linalg.cho_solve(  (self.BL,True) ,HM)  )[0,0]

	"""
	def Returnm_HN_m(self):
		m_HN_m =None
		HN = self.H - numpy.diag(1./self.N_array) 

		if self.verbose:
			pretime = time.time()
		for idx in range( HN.shape[1] ):
			left_mvec = numpy.copy(self.mvec[idx])
			right = None
			for H_i, mvec_i in zip( HN[idx,:], self.mvec):
				if right is None:
					right = H_i *  mvec_i
				else:
					right += H_i * mvec_i

			if m_HN_m is None:
				m_HN_m = left_mvec.T @ right
			else:
				m_HN_m += left_mvec.T @ right

		if self.verbose:
			posttime= time.time()

		if self.verbose:
			print('processing time of m_NH_m ', (posttime- pretime) )

		return -1.*m_HN_m[0,0]
	"""
	def ReturnmNm(self):
		mNm = None
		if self.is_noise_matrix:
			for ni_i,mvec_i in zip(self.NI_list, self.mvec):
				if mNm is None:
					mNm  = mvec_i.T @ ni_i @ mvec_i
				else:
					mNm += mvec_i.T @ ni_i @ mvec_i
		else:
			for ninv, mvec_i in zip(self.N_inv_array, self.mvec):
			#for ninv, mvec_i in zip(self.N_inv_array, self.tmpmvec):
				if mNm is None:
					mNm = ninv*(mvec_i.T@mvec_i)
				else :
					mNm += ninv*(mvec_i.T@mvec_i)
		return mNm[0,0]

	def ReturnmHm(self):
		mHm =None
		if self.verbose:
			pretime = time.time()
		for idx in range( self.H.shape[1] ):
			left_mvec = self.mvec[idx]
			#left_mvec = numpy.copy(self.mvec[idx])
			#left_mvec = sympy.Matrix(left_mvec)
			
			right = None
			#for H_i, mvec_i in zip( self.H.row(idx), self.mvec):
			for H_i, mvec_i in zip( self.H[idx,:], self.mvec):
				if right is None:
					right = H_i *  mvec_i
					#right = H_i/self.Delta *  mvec_i
				else:
					right += H_i * mvec_i
					#right += H_i/self.Delta *  mvec_i

			#right = sympy.Matrix( right )
			if mHm is None:
				#mHm = left_mvec.T * right
				mHm = left_mvec.T @ right
			else:
				#mHm += left_mvec.T *right
				mHm += left_mvec.T @right
		#return mHm[0,0].evalf()
		#mHm *= self.Delta
		if self.verbose:
			posttime= time.time()
		if self.verbose:
			print('processing time of mHm ', (posttime- pretime) )

		return -1.*mHm[0,0]

	def ReturnmNIAINIm(self):
		#return (-self.NIm.T * self.AI_NIm)[0,0].evalf()
		#return (-self.NIm.T @  self.AI_NIm)[0,0].evalf()
		#TODO
		#return (-self.NIm.T @  self.AI_NIm)[0,0] 
		#return ((-self.NIm.T @  self.AI_NIm)[0,0] + (-self.AI_NIm.T @ self.NIm )[0,0])/2.
		return (-self.NIm.T @  self.AI_NIm)[0,0]

	def ReturnmHAINIm(self):
		#return (2.*self.Hm.T * self.AI_NIm)[0,0].evalf()
		#return (2.*self.Hm.T @ self.AI_NIm)[0,0].evalf()
		#return 2.*(self.Hm.T @ self.AI_NIm)[0,0] 
		
		#return self.Delta*((self.Hm/self.Delta).T @ self.AI_NIm)[0,0] + self.Delta*(self.AI_NIm.T @ (self.Hm/self.Delta) )[0,0]
		#return (self.Hm.T @ self.AI_NIm)[0,0] + (self.NIm.T @ scipy.linalg.lu_solve((self.ALU, self.AP), self.Hm) )[0,0]
		return (self.Hm.T @ self.AI_NIm)[0,0] + (self.AI_NIm.T @ self.Hm )[0,0]

	"""
	def Returnm_HN_AINIm(self):
		D = self.dmtrx.D_matrix
		N_inv_array = 1./self.N_array
		NI = numpy.diag(N_inv_array)  

		#for param,value in zip(self.params, self.param_values):
		for param in self.params:
			#self.D = self.D.subs(param, value)
			D = D.subs(param, self.param_values[param.name])
			if self.verbose:
				print(str(param)+': ', value)
		#self.D = sympy.matrix2numpy(self.D, dtype = numpy.float64 ) 
		D = sympy.matrix2numpy(D, dtype = numpy.float64 ) 

		self.HN =  None
		self.HN = ( scipy.linalg.cho_solve((self.HL, True) , D.T @  NI) )
		self.HN = D @ self.HN
		self.HN = self.HN - numpy.identity(self.H.shape[1])
		self.HN = NI @  self.HN
		self.HNm = None
		for HN_i, mvec_i in zip( self.HN.sum(axis=0), self.mvec):
			if self.HNm is  None:
				self.HNm =  HN_i*numpy.eye(len(mvec_i)) @ mvec_i
			else:
				self.HNm += HN_i*numpy.eye(len(mvec_i)) @ mvec_i	
		return (self.AI_NIm.T @  self.HNm )[0,0]
	"""

	def ReturnmHBIHm(self):
		#return (self.Hm.T @ scipy.linalg.cho_solve( self.BL, self.Hm)[0,0].evalf()
		#return (self.Hm.T @ scipy.linalg.lu_solve( (self.BLU, self.BP ),self.Hm) )[0,0]
		#return self.Delta*(self.Hm.T @ scipy.linalg.lu_solve(  (self.BpDLU, self.BpDP ), self.Hm) )[0,0]
		#return ( (self.Hm.T/self.Delta) @ scipy.linalg.lu_solve(  (self.BpDLU, self.BpDP ), self.Hm) )[0,0]
		#return self.Delta * ( (self.Hm/self.Delta ).T @ scipy.linalg.lu_solve(  (self.BpDLU, self.BpDP ), self.Hm/self.Delta) )[0,0]
		#return ( (self.Hm/self.Delta ).T @ scipy.linalg.lu_solve(  (self.BpDLU, self.BpDP ), self.Hm) )[0,0]

		#TODO MODIFIED	
		#return (self.Hm.T @ scipy.linalg.lu_solve(  (self.BpDLU, self.BpDP ), self.Hm/self.Delta) )[0,0]
		#return (self.Hm.T @ scipy.linalg.lu_solve(  (self.BLU, self.BP ), (self.Hm ) ) )[0,0]
		return -(self.Hm.T @ scipy.linalg.cho_solve(  (self.BL, True), (self.Hm ) ) )[0,0]

	def ReturnDmHBIAINIm(self):
		#return (-2.*self.Delta*self.Hm.T * self.B.cholesky_solve( 
		"""
		return (-2.*self.Delta*self.Hm.T * self.B.LUsolve( 
			self.AI_NIm
		) )[0,0].evalf()
		"""
		#return (-2.*self.Delta * self.Hm.T @ scipy.linalg.cho_solve(self.BL, self.AI_NIm) )[0,0].evalf()
		#return (-2.* self.Hm.T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ), self.AI_NIm) )[0,0]
		#return (-2.* self.AI_NIm.T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.Hm ) )[0,0]
		#return -self.Delta*( self.AI_NIm.T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.Hm/self.Delta ) )[0,0] - self.Delta*( (self.Hm/self.Delta).T@ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ), self.AI_NIm) )[0,0]
		#return -self.Delta*( self.AI_NIm.T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.Hm/self.Delta ) )[0,0] - self.Delta*( scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.Hm/self.Delta ).T @ self.AI_NIm) [0,0]
		#return -( self.AI_NIm.T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.Hm ) )[0,0] - ( scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.Hm ).T @ self.AI_NIm) [0,0]
		#return -self.Delta*( self.AI_NIm.T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.BpD@(self.Hm/self.Delta)  ) )[0,0] - self.Delta*( scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.BpD@(self.Hm/self.Delta) ).T @ self.AI_NIm) [0,0]
		#return -self.Delta*(  self.Hm.T @scipy.linalg.lu_solve( (self.BLU, self.BP ), self.AI_NIm ) )[0,0] - self.Delta*( scipy.linalg.lu_solve( (self.BLU, self.BP ), self.AI_NIm ).T @ self.Hm) [0,0]
		#return (-2.* (self.Hm/self.Delta).T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ), self.Delta*self.AI_NIm ) )[0,0]

		#TODO MODIFIED	
		##return -(  self.Hm.T @scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ), self.AI_NIm ) )[0,0] - ( self.AI_NIm.T @  scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),  self.Hm)) [0,0]
		##return -2.*self.Delta*( self.AI_NIm.T @ scipy.linalg.lu_solve( (self.BpDLU, self.BpDP ),self.Hm/self.Delta ) )[0,0]
		#return -2*self.Delta*( self.Hm.T @  scipy.linalg.lu_solve( (self.BLU, self.BP ), self.AI_NIm )) [0,0]
		#return -(  self.Hm.T @scipy.linalg.lu_solve( (self.BLU, self.BP ), self.Delta*self.AI_NIm ) )[0,0] - ( (self.Delta*self.AI_NIm).T @  scipy.linalg.lu_solve( (self.BLU, self.BP ),  self.Hm)) [0,0]
		##return -2*(  self.Hm.T @scipy.linalg.lu_solve( (self.BLU, self.BP ), self.Delta*self.AI_NIm ) )[0,0]
		#return -(  self.Hm.T @scipy.linalg.lu_solve( (self.BLU, self.BP ), self.Delta*self.AI_NIm ) )[0,0] - ( (self.NIm).T @ (self.Delta* scipy.linalg.lu_solve( (self.ALU, self.AP),scipy.linalg.lu_solve( (self.BLU, self.BP ),  self.Hm)) ) ) [0,0]
		#return -self.Delta*(  self.Hm.T @scipy.linalg.lu_solve( (self.BLU, self.BP ), self.AI_NIm ) )[0,0] - self.Delta*( self.AI_NIm.T @  scipy.linalg.lu_solve( (self.BLU, self.BP ),  self.Hm)) [0,0]
		return self.Delta*(  self.Hm.T @scipy.linalg.cho_solve( (self.BL, True ), self.AI_NIm ) )[0,0] + self.Delta*( self.AI_NIm.T @  scipy.linalg.cho_solve( (self.BL, True ),  self.Hm)) [0,0]

	def ReturnDmNIAIAINIm(self):
		
		#TODO MODIFIED	
		#return -1./self.Delta * ( (self.Delta *self.AI_NIm).T @ (self.Delta *self.AI_NIm) )[0,0]
		return -1.*self.Delta * ( self.AI_NIm.T @ (self.AI_NIm) )[0,0]
		#return -1.* ( self.AI_NIm.T @ (self.Delta *self.AI_NIm) )[0,0]

	def ReturnD2mNIAIBIAINIm(self):
		#return pow(self.Delta,1) * (self.AI_NIm.T @ scipy.linalg.lu_solve((self.BpDLU, self.BpDP), self.BpD@self.AI_NIm) )[0,0]
		#return pow(self.Delta,2) * (self.AI_NIm.T @ scipy.linalg.lu_solve((self.BLU, self.BP), self.B@self.AI_NIm) )[0,0]
		#return  (self.AI_NIm.T @ (self.Delta* scipy.linalg.lu_solve((self.BpDLU, self.BpDP), self.AI_NIm) ) )[0,0]
		#return (self.AI_NIm.T @ scipy.linalg.lu_solve((self.BpDLU, self.BpDP), self.Delta * self.AI_NIm) )[0,0]
		#return 1./self.Delta *( (self.Delta * self.AI_NIm).T @ scipy.linalg.lu_solve((self.BpDLU, self.BpDP), self.Delta * self.AI_NIm) )[0,0]
		
		#return pow(self.Delta,1) * (self.AI_NIm.T @ scipy.linalg.lu_solve((self.BpDLU, self.BpDP), self.AI_NIm) )[0,0]
		#return ((self.Delta*self.AI_NIm).T @ scipy.linalg.lu_solve((self.BLU, self.BP), self.Delta*self.AI_NIm) )[0,0]
		#TODO MODIFIED	
		#return self.Delta*(self.AI_NIm.T @ scipy.linalg.lu_solve((self.BpDLU, self.BpDP), self.AI_NIm) )[0,0]
		#return pow(self.Delta,2) * (self.AI_NIm.T @ scipy.linalg.lu_solve((self.BLU, self.BP), self.AI_NIm) )[0,0]
		return -pow(self.Delta,2) * (self.AI_NIm.T @ scipy.linalg.cho_solve((self.BL, True), self.AI_NIm) )[0,0]

	def ReturnSN(self):
		S = numpy.kron(numpy.ones( (self.N_array.shape[0], self.N_array.shape[0])  ), self.S0_CMB )
		N = numpy.diag(self.N_array)
		N = numpy.kron( N, numpy.eye( self.size )  )
		if self.verbose:
			print('S : '+str(S))
			print('N: '+str(N))
		SN = S+N
		#return numpy.log( 2*numpy.pi*float(SN.det() ))
		sign,det =numpy.linalg.slogdet(SN)
		
		#return sign*det
		return det

	def ReturnlnS0(self):
		sign,det = numpy.linalg.slogdet(self.S0_CMB_L)
		#return 2*sign*det
		return 2*det

	def ReturnlnC(self):
		sgn,ldet = numpy.linalg.slogdet(self.A - self.S0_CMB_I - self.Delta* numpy.identity(self.size) )
		return self.ReturnlnS0() + self.ReturnlnB()-sgn*ldet

	def ReturnlnS(self):
		##combine all the noise
		n_comb =0.0
		for noise in self.N_inv_array:
			n_comb += noise
		Ncomb = numpy.eye(self.size)*1./n_comb
		SN =  self.S0_CMB + Ncomb
		sign,det =numpy.linalg.slogdet(SN)
		#return sign*det
		return det

		"""
		#detL = self.S0_CMB_L.diagonal().prod()
		#lndetL_old = numpy.log(self.S0_CMB_L.diagonal()).sum()

		lndetL = numpy.log( abs(self.S0_CMB_LU.diagonal() ) ).sum()
		#if lndetL_old != lndetL:
			#print((lndetL-lndetL_old)/lndetL_old)
		if self.verbose:
			#print('L diagonal = ', self.S0_CMB_L.diagonal())
			print('LU diagonal = ', self.S0_CMB_LU.diagonal() )
		#return 2.*float(self.N_array.shape[0]) * numpy.log(detL )
		return 2.*float(self.N_array.shape[0]) * lndetL
		"""
	def ReturnlnAI(self):
		"""
		sign,det =numpy.linalg.slogdet(self.ALU)
		return - sign*det
		"""
		sign,det = numpy.linalg.slogdet(self.AL)
		#return - 2*sign*det
		return - 2*det
	def ReturnlnB(self):
		sign,det = numpy.linalg.slogdet(self.BL)
		return 2*det

	def ReturnlnDNID(self):
		sign,det = numpy.linalg.slogdet(self.DNIDL)
		#return 2*sign*det*self.size
		#return 2*det*self.size
		return 2 * det
		#sign,det = numpy.linalg.slogdet(self.DNIDLU)
		#return sign*det*self.size

	def ReturnlnDT_SpNI_D(self):
		sign,det = numpy.linalg.slogdet(self.DT_SpNI_DL)
		return 2*det

	def ReturnxRTerm(self):
		xR = self.meanxR
		for param in self.params:
			if 'x^R' in param.name:
				xR = self.param_values[param.name]
		return pow( ( xR - self.meanxR )  /self.sigmaxR , 2 )
	def ReturnTdTerm(self):
		Td = self.meanTd	
		for param in self.params:
			if 'T_d1' in param.name:
				Td = self.param_values[param.name]
		return pow( ( Td - self.meanTd)  /self.sigmaTd,2)
	
	def ReturnChiSquare(self):
		self.valid = self.CalcInOneLoop()
		if not self.valid:
			return numpy.finfo(numpy.float64).max*1.e-3
		mNIAINIm = self.ReturnmNIAINIm()

		if not self.is_noise_matrix:
			mNm =  self.ReturnmNm()
		else:
			mNm = 0.0
		MHM = self.ReturnMHM()
		MHBIHM = self.ReturnMHBIHM()
		TdTerm = 0.0
		if self.withTdPrior:
			TdTerm = self.ReturnTdTerm()
		if self.withxRPrior:
			TdTerm = self.ReturnxRTerm()
		return mNm + mNIAINIm + MHM + MHBIHM + TdTerm


	def ReturnLikelihood(self):
		
		#mHAINIm = self.ReturnmHAINIm()
		#mHBIHm = self.ReturnmHBIHm()
		#DmHBIAINIm = self.ReturnDmHBIAINIm()
		#DmNIAIAINIm = self.ReturnDmNIAIAINIm()
		#D2mNIAIBIAINIm = self.ReturnD2mNIAIBIAINIm()
		#mHm =  self.ReturnmHm()
		#lnS = self.ReturnlnS()

		#lnAI =self.ReturnlnAI()

		nume = self.ReturnChiSquare()
		lnS0 = self.ReturnlnS0()
		denomi = 0.
		lnB =  self.ReturnlnB()
		lnDNID = self.ReturnlnDNID()
		"""
		if self.is_noise_matrix:
			lnAI = self.ReturnlnAI()
			lnDT_SpNI_D = self.ReturnlnDT_SpNI_D()
			denomi = lnS0 + lnAI + lnDT_SpNI_D
		else:
			lnB =  self.ReturnlnB()
			lnDNID = self.ReturnlnDNID()
			denomi = lnS0 + lnB +  lnDNID
		"""

		denomi = lnS0 + lnB +  lnDNID
		#denomi = lnS0 + lnB

		#lnSN = self.ReturnSN()
		#return  self.ReturnmNIAINIm() + self.ReturnmHAINIm() + self.ReturnmHBIHm()+ self.ReturnDmHBIAINIm() + self.ReturnDmNIAIAINIm()+ self.ReturnD2mNIAIBIAINIm() 	+ self.ReturnmHm() + self.ReturnSN()
		#return  self.ReturnmNIAINIm() + self.ReturnmHAINIm() + self.ReturnmHBIHm()+ self.ReturnDmHBIAINIm() + self.ReturnDmNIAIAINIm()+ self.ReturnD2mNIAIBIAINIm() 	+ self.ReturnmHm() + self.ReturnlnS()
		#return mNIAINIm + mHAINIm + mHBIHm + DmHBIAINIm + DmNIAIAINIm + D2mNIAIBIAINIm + mHm + lnS
		#return m_HN_AINIm + mHAINIm/2. + mHBIHm + DmHBIAINIm + DmNIAIAINIm + D2mNIAIBIAINIm + mHm + lnS
		#return mNIAINIm + mHAINIm + mHBIHm + DmHBIAINIm + DmNIAIAINIm + D2mNIAIBIAINIm + m_HN_m + lnS
		#return mNIAINIm + mHAINIm + mHBIHm + DmHBIAINIm + DmNIAIAINIm + D2mNIAIBIAINIm + mHm + lnSN + mNm
		#return mNIAINIm + mHAINIm + mHBIHm + DmHBIAINIm + DmNIAIAINIm + D2mNIAIBIAINIm + mHm + lnS + mNm
		#return mNIAINIm + lnS + mNm +  MHM + MHBIHM
		#return mNIAINIm + lnSN + mNm +  MHM + MHBIHM
		#return mNIAINIm + lnS + mNm +  MHM + MHBIHM +lnB+lnAI + lnDNID
		#return mNIAINIm + lnS0 + mNm +  MHM + MHBIHM +lnB + lnDNID
		#return mNIAINIm + lnS0 + mNm +  MHM + MHBIHM - lnAI
		return denomi + nume
	
	def IterateMinimize(self):
		#tmp_r = self.inits['r'][0]
		self.lh = 1.0e10
		tmp_r = 100
		pre_lh = numpy.inf
		pre_r = numpy.inf
		count = 0
		self.param_errors = {}
		for key in self.inits.keys():
			self.param_errors[key] = ( self.inits[key][1][1] - self.inits[key][1][0])/2.
		#while( abs(tmp_r - pre_r) > 1.0e-10 ):
		while( (( pre_lh - self.lh ) > 1.0e-2 ) or abs(pre_r - tmp_r) > 1.0e-5 ):
			#fit without r
			self.fg_valid = False
			while(not self.fg_valid):
				fmin,fparam = self.ReturnMinimize(False, False)
				break
			pcount = 0
			for param in self.params:
				if 'r' in param.name:
					continue
				else:
					#self.inits[param.name][0] = fparam[pcount]['value']
					self.param_values[param.name] = fparam[pcount]['value']
					self.param_errors[param.name] = fparam[pcount]['error']
					pcount +=1
			if self.verbose:
				print(self.inits)
			self.r_valid = False
			while(not self.r_valid):
				fmin,fparam = self.ReturnMinimize(True,True)
				break
			self.param_values['r'] = fparam[0]['value']
			self.param_errors['r'] = fparam[0]['error']
			pre_r = tmp_r 
			tmp_r = fparam[0]['value']
			#self.r_minos = fparam[0]['merror']
			if self.verbose:
				print(self.param_values)
			#pre_r = tmp_r
			pre_lh = self.lh
			#tmp_r = self.inits['r'][0]
			#tmp_r = self.param_values['r']
			#tmp_lh = self.MinimizeOnlyR([ self.param_values['r'] ])
			self.lh = fmin.fval
			if self.r_verbose:
				print(self.lh, tmp_r)
			if count>=20:
				print('iteration over 20')
				break
			count +=1
		return self.param_values, self.param_errors

	def ReturnMinimize(self, isfit_r = True , isonly_r =False):
		parameter_initial = []
		limits = []
		err_params = []
		for param in self.params:
			if 'r' in param.name:
				if not isfit_r:
					continue
				else:
					parameter_initial.append(self.inits[param.name][0])
					#parameter_initial.append(self.param_values[param.name])
					limits.append(self.inits[param.name][1])
					err_params.append(0.5)
			else:
				if isonly_r:
					continue
				else:
					#parameter_initial.append(self.inits[param.name][0])
					parameter_initial.append(self.param_values[param.name])
					limits.append(self.inits[param.name][1])
					err_params.append(0.5)

		"""
		for param, limit in zip(self.params, self.limits):
			if not isfit_r and 'r' in str(param) :
				continue
			else:
				parameter_initial.append(self.inits[symbol.name])
				limits.append(limit)
				err_params.append(0.5)
		"""
		if self.verbose:
			print(parameter_initial, limits)
		if isfit_r:
			if not isonly_r:
				self.m = Minuit(
					self.MinimizeWithR , 
					parameter_initial
				)
				self.m.limits = limits
				self.m.errordef = 1
				self.m.print_level = 0
				self.m.strategy = 2

			else:
				self.m = Minuit(
					self.MinimizeOnlyR , 
					parameter_initial
				)
				self.m.limits = limits
				self.m.errordef = 1
				self.m.print_level = self.r_verbose
				self.m.strategy = 2

		else:
			self.m = Minuit(
				self.MinimizeWithoutR , 
				parameter_initial					)
			self.m.limits = limits
			self.m.errordef = 1
			self.m.print_level =  self.r_verbose
			self.m.strategy = 2
	
		if not isfit_r:
			self.m.migrad()
			fmin = self.m.fmin
			fparam = []
			#for value,error in zip(self.m.values, self.m.errors):
			for p in self.m.params:
				fparam.append({'value': p.value, 'error':p.error})
			self.fg_valid = self.m.valid
			return fmin,fparam
		else:
			"""
			self.m.scan()
			if self.r_verbose:
				print(self.m.params[0].value ,self.m.fmin.fval)
			"""
			if isfit_r and not self.migrad:
				self.m.scipy()
			else:
				self.m.migrad()
			fmin = self.m.fmin
			if self.r_verbose:
				print(self.m.params[0].value ,self.m.fmin.fval)
			self.r_valid = self.m.valid
			if self.m.valid:
				self.m.minos()
			params = self.m.params
			fparam = []
			for p in params:
				fparam.append({'value': p.value, 'error':p.error})
			"""
			self.m.scipy()
			sci_fmin = self.m.fmin
			sci_valid = self.m.valid
			if sci_valid:
				self.m.minos()
			sci_params = self.m.params

			if mig_fmin.fval <= sci_fmin.fval:
				self.r_valid = mig_valid
				fmin = mig_fmin
				fparam = []
				for p in mig_params:
					fparam.append({'value': p.value, 'error':p.error, 'merror':[p.merror]})
			else:
				self.r_valid = sci_valid
				fmin = sci_fmin
				fparam = []
				for p in sci_params:
					fparam.append({'value': p.value, 'error':p.error, 'merror':[p.merror]})
			"""
			return fmin,fparam
				


		"""
		if isfit_r and self.m.valid:
			self.m.minos()
		self.r_valid = self.m.valid
		"""
	
	def MinimizeOnlyR(self,params):
		for idx, param in enumerate(self.params):
			if not 'r' in param.name:
				pass
				#self.param_values[param.name] = self.inits[param.name][0]
			else:
				self.param_values[param.name] = params[0]
		return self.ReturnLikelihood()

	def MinimizeWithoutR(self, params):
		count =0
		##TODO##
		for idx,param in enumerate(self.params):
			if 'r' in param.name :
				pass
				#self.param_values[param.name] = self.inits[param.name][0]
			else:
				self.param_values[param.name] = params[count]
				count += 1
		if self.verbose:
			print(self.params, self.param_values[param.name])
		#return self.ReturnLikelihood()
		return self.ReturnChiSquare()

	def MinimizeWithR(self, params):
		for idx,param in enumerate(self.params):
			self.param_values[param.name] = params[idx]
		#print(self.params, self.param_values[param.name])
		return self.ReturnLikelihood()

	def ReturnLikelihoodValue(self, values):
		self.SetParameters(values)
		return self.ReturnLikelihood()

	def ReturnLikelihoods(self,param_lists):
		return [ self.ReturnLikelihoodValue(param) for param in param_lists]

	def CalcInOneLoop(self):
		self.valid = True
		#self.CalcTmpVec()
		#self.CalcNIm()#TODO
		self.CalcCMB0Inverse()
		if not self.valid:
			return False
		self.CalcA()
		if not self.valid:
			return False
		self.CalcAI_NIm()
		self.CalcMeanVec()
		if self.is_noise_matrix:
			self.CalcH_matrix()
			#self.CalcDTNIDc()
			#self.CalcDTNIM()
			self.CalcDTNID_I_DTNIM()
			self.CalcDcTNID_DTNID_I_DTNIM()
			self.CalcDelta()

		else:
			self.CalcH()
		if not self.valid:
			return False 


		self.CalcB()
		if not self.valid:
			return False
		#self.CalcAm()
		#self.CalcMiddle()
		if self.is_noise_matrix:
			self.CalcBIDcTNID_DTNID_I_DTNIM()
		return True
	
	def is_pos_def(self, A):
		if numpy.array_equal(A, A.T):
			try:
				numpy.linalg.cholesky(A)
				return True
			except numpy.linalg.LinAlgError:
				return False
		else:
			print('input matrix is not symmetric')
			return False

	def upper_triangular_to_symmetric(self, ut):
		
		n = ut.shape[0]
		try:
			inds = self.inds_cache[n]
		except KeyError:
			inds = numpy.tri(n, k=-1, dtype=numpy.bool)
			self.inds_cache[n] = inds
		ut[inds] = ut.T[inds]

	def fast_positive_definite_inverse(self, m):
		cholesky, info = lapack.dpotrf(m)
		if info != 0:
			raise ValueError('dpotrf failed on input {}'.format(m))
		inv, info = lapack.dpotri(cholesky)
		if info != 0:
			raise ValueError('dpotri failed on input {}'.format(cholesky))
		self.upper_triangular_to_symmetric(inv)
		return inv

def main():
	"""
	main function of deltamap.py

	"""
	return 0

if __name__ == '__main__':
	sys.exit(main())
