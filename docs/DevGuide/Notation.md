\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Notation Used by SpECTRE {#spectre_notation_general}

# Discontinuous Galerkin Notation {#dg_notation}

- The logical coordinates in the element are
  \f$\mathbf{\xi}=\{\xi,\eta,\zeta\}\f$.
- The Jacobian matrix is defined as
  \f{align*}{
      \mathbf{J}=\frac{\partial \mathbf{x}}{\mathbf{\xi}}
  \f}
  where \f$\mathbf{x}\f$ are the coordinates mapped to by coordinate map.
  The determinant of the Jacobian matrix is denoted by \f$J\f$.
- We use \f$s\f$ and \f$t\f$ to be grid point indices. That is, \f$U_s\f$ is
  \f$U\f$ at the grid point \f$s\f$. For tensor product bases we index the
  individual irreducible topologies by subscripting \f$s\f$. For example,
  \f$s_1\f$ would be the \f$\xi\f$-dimension in the reference element.
- We use a nodal basis of Lagrange interpolating polynomials
  \f{align*}{
      \ell_{s_i}(\xi)=\prod^{N}_{\substack{t_i=0\\ t_i\ne s_i}}
      \frac{\xi - \xi_{t_i}}{\xi_{s_i}-\xi_{t_i}},
  \f}
  where \f$N\f$ is the order/degree of the basis in each logical direction.
- A DG scheme with an \f$N^{\mathrm{th}}\f$ order basis is referred to as a
  \f$P_N\f$ scheme and converges at order \f$\mathcal{O}\left(\Delta
  x^{N+1}\right)\f$ where \f$\Delta x\f$ is the 1d size of the element.
- The unit normal vector to the element is denoted by \f$n_i\f$.
- The numerical flux boundary correction term which includes the normal vectors
  is denoted by \f$G(F^{i,+},n_i^+,u^+,F^{i,-},n_i^-,u^-)\f$ where the \f$+\f$
  and \f$-\f$ denote the values on the outer and inner side of the mortar,
  respectively. To give a concrete example, for the Rusanov flux we have
  \f{align*}{
      G^{\mathrm{Rusanov}}=
      \frac{1}{2}\left(F^{i,+}n_i^+-F^{i,-}n_i^-\right)
      -\frac{C}{2}\left(u^+-u^-\right)
  \f}

# General Relativity and (Magneto)hydrodynamics{#gr_hydro_notation}

We use the following conventions and notation:
- In general we work in units where \f$G=c=1\f$.
- Latin letters at the beginning of the alphabet, \f$a,b,c,\ldots\f$ are
  spacetime indices ranging from 0 to 3.
- Latin letters \f$i,j,k,\ldots\f$ are spatial indices ranging from 1 to 3.

The notation for general relativity quantities used is:
- The Einstein summation convention is used where repeated indices are summed
  over.
- The spacetime metric is denoted by \f$g_{ab}\f$ with signature
  \f$(-,+,+,+)\f$, and its determinant by \f$g\f$.
- The lapse and shift are denoted by \f$\alpha\f$ and \f$\beta^i\f$,
  respectively.
- The spatial metric and its determinant are denoted by \f$\gamma_{ij}\f$ and
  \f$\gamma\f$, respectively.
- We write the line element as:
  \f{align*}{
      ds^2=g_{ab}dx^a dx^b=-\alpha^2 dt^2 +
      \gamma_{ij}
      \left(dx^i + \beta^i dt\right)\left(dx^j + \beta^j dt\right).
  \f}
- The unit timelike normal to the hypersurface is
  \f$n^a=(1/\alpha,-\beta^i/\alpha)\f$ and \f$n_a=(-\alpha,0,0,0)\f$.
- The extrinsic curvature is denoted by \f$K_{ij}\f$ and its trace by
  \f$K\f$. We use the sign convention
  \f{align*}{
      K_{ab}=-\frac{1}{2}\mathcal{L}_n g_{ab}
  \f}
- For the generalized harmonic evolution system \f$\Pi_{ab}=-n^c\partial_c
  g_{ab}\f$ and \f$\Phi_{iab}=\partial_i g_{ab}\f$.

The notation for (magneto)hydrodynamics used is:
- The fluid pressure, rest mass density, specific internal energy, and specific
  enthalpy are denoted by \f$p\f$, \f$\rho\f$, \f$\epsilon\f$, and \f$h\f$,
  respectively.
- The 4-velocity of the fluid is \f$u^a\f$ and \f$u^a u_a=-1\f$.
- The spatial velocity of the fluid as measured by an observer at rest in the
  spatial hypersurfaces (“Eulerian observer”) is
  \f{align*}{
      v^i=\frac{1}{\alpha}\left(\frac{u^i}{u^0}+\beta^i\right)
  \f}
  with corresponding Lorentz factor
  \f{align*}{
      W=-u^a n_a=\alpha u^0=\frac{1}{\sqrt{1-\gamma_{ij}v^i v^j}}
  \f}
- The Faraday tensor is denoted by \f$F^{ab}\f$ and its dual is given by
  \f$^{*}\!F^{ab}=\frac{1}{2}\epsilon^{abcd}F_{cd}\f$ where
  \f$\epsilon^{abcd}\f$ is the Levi-Civita tensor.
- The magnetic field in the frame comoving with the fluid is
  \f{align*}{
      b^a = ^{*}\!F^{ab}u_b
  \f}
- The electric and magnetic fields as measured by an Eulerian observer are given
  by
  \f{align*}{
      E^i=&F^{ia}n_a=\alpha F^{0i} \\
      B^i=&^{*}\!F^{ia}n_a=\alpha^{*}\!F^{0i}.
  \f}
- \f$b^a\f$ can be written in terms of \f$E^i\f$ and \f$B^i\f$ as follows:
  \f{align*}{
      b^0=&\frac{W}{\alpha}B^i v_i\\
      b^i=&\frac{B^i+\alpha b^0u^i}{W}
  \f}
  while \f$b^2=b^i b_i\f$ is given by
  \f{align*}{
      b^2=\frac{B^2}{W}+\left(B^i v_i\right)
  \f}
- The magnetic pressure is \f$p_{\mathrm{mag}}=b^2/2\f$
- The primitive variables are \f$\rho\f$, \f$\epsilon\f$, \f$v_i\f$, \f$B^i\f$,
  and the divergence cleaning field \f$\Phi\f$.
- The conserved variables are:
  \f{align*}{
      D=&\rho W\\
      S_i=&(\rho h)^{*}W^2 v_i - \alpha b^0 b_i \\
      \tau =& (\rho h)^{*}W^2 - p^* - \left(\alpha b^0\right)^2-\rho W \\
      B^i =& B^i \\
      \Phi =& \Phi,
  \f}
  where
  \f{align*}{
      (\rho h)^* =& \rho h + b^2=\rho h + 2 p_{\mathrm{mag}} \\
      p^* = & p + \frac{b^2}{2}=p+p_{\mathrm{mag}}.
  \f}
  We denote the conserved variables multiplied by \f$\sqrt{\gamma}\f$ with a
  tilde on top. For example, \f$\tilde{D}=\sqrt{\gamma}D\f$.
