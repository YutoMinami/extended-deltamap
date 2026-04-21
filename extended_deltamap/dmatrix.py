import numpy
import sympy


class DMatrix:
    """Build symbolic mixing matrices for foreground components."""

    def __init__(self, verbose=False):
        """Initialize an empty symbolic D-matrix builder.

        Args:
            verbose: Whether to print intermediate symbolic matrices.
        """
        self.dim_params = 0
        self.d_params = []
        self.d_funcs = []
        self.freqs = None
        self.width = None
        self.n_freqs = None
        self.D_matrix_template = []
        self.D_matrix = None
        self.gnu = None
        self.verbose = verbose
        pass

    def SetGnu(self, gnu):
        """Store an optional bandpass correction term.

        Args:
            gnu: Symbolic or numeric bandpass factor.
        """
        self.gnu = gnu

    def AddD(self, f_D):
        """Register one foreground scaling law.

        Args:
            f_D: Symbolic frequency scaling function containing ``nu`` and any
                fit parameters.
        """
        # count params
        symbols = f_D.free_symbols
        params = []
        self.dim_params += 1
        for param in symbols:
            if "nu" not in param.name:
                self.dim_params += 1
                params.append(param)
        self.d_params.append(params)
        self.d_funcs.append(f_D)

    def SetFreqs(self, freqs, width):
        """Set the observing frequencies used to evaluate the matrix.

        Args:
            freqs: Frequency channels in GHz.
            width: Optional bandwidth values carried alongside the frequencies.
        """
        self.freqs = numpy.asarray(freqs)
        self.width = numpy.asarray(width)
        self.n_freqs = len(freqs)

    def _build_template_terms(self, diff_param=None):
        """Build one fresh list of symbolic template terms for the current settings.

        Args:
            diff_param: Optional parameter-name fragment used to keep only one
                subset of first-derivative columns.

        Returns:
            A newly built list of symbolic basis terms.
        """
        template_terms = []
        for func, params in zip(self.d_funcs, self.d_params):
            template_terms.append(func)
            for param in params:
                if diff_param is not None and diff_param not in param.name:
                    continue
                template_terms.append(sympy.simplify(sympy.diff(func, param)))
            # Future second-order work should append terms here in a stable
            # order, including mixed derivatives when models have >1 parameter.
        return template_terms

    def _evaluate_template_terms(self, template_terms):
        """Evaluate symbolic template terms at each configured frequency."""
        l_Dmatrix = []
        for nu in self.freqs:
            nu_D = [func.subs("nu", nu) for func in template_terms]
            l_Dmatrix.append(nu_D)
        return l_Dmatrix

    def DiffD(self):
        """Populate the template with each component and its parameter derivatives."""
        self.D_matrix_template = self._build_template_terms()

    def PrepareOneDiffDMatrix(self, diff_param):
        """Build a matrix that includes derivatives for one selected parameter.

        Args:
            diff_param: Parameter-name fragment used to choose which derivative
                columns to include.
        """
        self.D_matrix_template = self._build_template_terms(diff_param=diff_param)
        l_Dmatrix = self._evaluate_template_terms(self.D_matrix_template)
        if self.verbose:
            print(self.dim_params, self.n_freqs)
            print((l_Dmatrix))
        self.D_matrix = sympy.Matrix(l_Dmatrix)

    def PrepareUniformDMatrix(self):
        """Build a matrix with component amplitudes only, without derivatives."""
        self.D_matrix_template = []
        for func, params in zip(self.d_funcs, self.d_params):
            self.D_matrix_template.append(func)
        l_Dmatrix = self._evaluate_template_terms(self.D_matrix_template)
        if self.verbose:
            print(self.dim_params, self.n_freqs)
            print((l_Dmatrix))
        self.D_matrix = sympy.Matrix(l_Dmatrix)

    def PrepareDMatrix(self):
        """Build the full symbolic D matrix including parameter derivatives."""
        self.DiffD()
        l_Dmatrix = self._evaluate_template_terms(self.D_matrix_template)
        if self.verbose:
            print(self.dim_params, self.n_freqs)
            print((l_Dmatrix))
        # self.D_matrix =sympy.Matrix(self.n_freqs, self.dim_params, l_Dmatrix)
        self.D_matrix = sympy.Matrix(l_Dmatrix)

    def CalcDMatrix(self):
        """Placeholder for future explicit D-matrix evaluation logic."""
        # calculate n_freqs of Ds
        # calculate n_params of Ds
        pass
