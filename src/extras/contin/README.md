# CONTIN input file

## Parameters
See contin-manual1.pdf.
For boolean flags, 1=TRUE and 0=FALSE

- **LAST**: if TRUE, CONTIN stops after analysis
- **GMNMX(1)**: first grid point in the quadrature
- **GMNMX(2)**: last point in the quadrature
- **IWT**:  **1** = unweighted analysis; **2** = error proportional to $\sqrt{y(t_k)}$, i.e., Poisson statistics; **3** = error proportional to $|y(t_k)|$, i.e., constant relative error; **4** = input wk in Card set 7; **5** = computing $w_k$ in USERWT
- **NERFIT**: number of residuals used to compute ERRFIT
**NINTT**: number of intervals for $t_k$. **THIS SHOULD ALWAYS BE $\le 0$!!**
- **NLINF**: NL in Eq. 3.1-2
- **IFORMY**: Fortran FORMAT specification enclosed in parentheses for $y_k$ in Card Set 6
- **IFORMT**: Fortran FORMAT specification enclosed in parentheses for $t_k$ in Card Set 5b
- **IFORMW**: Fortran FORMAT specification enclosed in parentheses for $w_k$ in Card Set 7
- **NG**: Number of quadrature grid points, if Eq.3.1-2 is solved by quadrature. $N_x$ if Eq.3.1-1 is solved directly
- **DOUSNQ**: if TRUE, USERNQ is to be called to specify inequality constraints
- **USERNQ**: sets the inequality constraints in Eq.3.1-4
- **NONNEG**: if TRUE, constrains $s(g_m)$ in Eq.3.1-3 to be non-negative
- **NORDER**: choice of the regularizors. **$< 0$** for calling USERRG to set a special user-defined regularizor; **0-5** for setting the regularizor (sum in Eq.3.2.2-1) to be the sums of the squares of the nth differences of the $N_g-n$ sets (similar to order of derivative). **0** is $\sum_j x_j^2$, **2** is sum of second derivative squared (smooth solution)
- **IPLRES(2)**: controls when the weighted residuals will be plotted. **0** = never; **1** = only after peak-constrained solution; **2** = also after the CHOSEN SOLUTION; **3** = after every solution
- **IPLFIT(2)**: same as IPLRES, except that it controls when the plots of the fit to the data will be made.
- **RUSER(1)**: $s(\lambda_1)$
- **RUSER(2)**: $s(\lambda_{N_g})$
- **RUSER(3)**: noise level
- **RUSER(6)**: integral of $s(\lambda)$
- **RUSER(10)**: **0** for not changing $y_k$ (<span style="color:red">**ONLY THIS SHOULD BE USED FOR DDM ANALYSIS**</span>); **$< 0$** for replacing $y_k$ with $\sqrt{y_k}$; **$> 0$** for replacing $y_k$ with $\sqrt{y_k/R_{10}-1}$
- **RUSER(15)**: medium refractive index.
- **RUSER(16)**: wavelength illumination source (in nm). If $R_{16}=0$, $R_{20}$ is not computed and $R_{21}$ is set to 0.
- **RUSER(17)**: scattering angle (in degrees).
- **RUSER(18)**: absolute temperature.
- **RUSER(19)**: viscosity (in cP).
- **RUSER(20)**: scattering vector (in $\mathrm{cm}^{-1}$). Computed from $R_{15}$, $R_{16}$, and $R_{17}$.
- **RUSER(24)**: wall thickness of hollow spheres (in cm).
- **IUSER(10)**:  **1** = $s(\lambda)$ is weight fraction molecular weight distribution ($R_{23}=1$, $R_{22}=R_{18} R_{20}^2$, so that $D=R_{18} \lambda^{R_{22}}$); **2** = $s(\lambda)$ is diffusion coefficient distribution ($R_{23}=0$, $R_{22}=1$, $R_{21}=R_{20}^2$); **3** = $s(\lambda)$ is weight fraction radius distribution (in cm) of spheres satisfying the Stokes-Einstein relation ($R_{23}=3$, $R_{22}=-1$, $R_{21} = k_B R_{18} R_{20}^2/(0.06 \pi R_{19})$); **4** = generic case where $R_{21}$, $R_{22}$, and $R_{23}$ are set by the user
- **LUSER(3)**: **0** = do not use form factors (i.e., $f_m=1$); **1** = use Rayleigh-Debye form factors for hollow spheres with $R_{24}$ wall thickness (in cm; if $R_{24} \le 0$, the form factors for solid spheres are computed). An $I_{18}>0$ causes the squared form factor to be averaged over $2 I_{18} + 1$ equally spaced points on the interval centered at the grid point and extending halfway to its nearest neighbors (if form factor rapidly oscillates). Default $I_{18}=50$ is recommended.

Kernels are of the form
$F(\lambda_m, t_k) = f_m^2 \lambda_m^{R_{23}} \exp(-R_{21} t_k \lambda_m^{R_{22}})$.
