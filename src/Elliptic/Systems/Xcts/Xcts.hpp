// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Documents the `Xcts` namespace

#pragma once

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the Extended Conformal Thin Sandwich (XCTS)
 * decomposition of the Einstein constraint equations
 *
 * The XCTS equations
 *
 * \f{align}
 * \bar{D}^2 \psi - \frac{1}{8}\psi\bar{R} - \frac{1}{12}\psi^5 K^2 +
 * \frac{1}{8}\psi^{-7}\bar{A}_{ij}\bar{A}^{ij} &= -2\pi\psi^5\rho
 * \\
 * \bar{D}_i(\bar{L}\beta)^{ij} - (\bar{L}\beta)^{ij}\bar{D}_i
 * \ln(\bar{\alpha}) &= \bar{\alpha}\bar{D}_i\left(\bar{\alpha}^{-1}\bar{u}^{ij}
 * \right) + \frac{4}{3}\bar{\alpha}\psi^6\bar{D}^j K + 16\pi\bar{\alpha}
 * \psi^{10}S^j
 * \\
 * \bar{D}^2\left(\alpha\psi\right) &=
 * \alpha\psi\left(\frac{7}{8}\psi^{-8}\bar{A}_{ij}\bar{A}^{ij}
 * + \frac{5}{12}\psi^4 K^2 + \frac{1}{8}\bar{R}
 * + 2\pi\psi^4\left(\rho + 2S\right)\right)
 * - \psi^5\partial_t K + \psi^5\beta^i\bar{D}_i K
 * \\
 * \text{with} \quad \bar{A} &= \frac{1}{2\bar{\alpha}}
 * \left((\bar{L}\beta)^{ij} - \bar{u}^{ij}\right) \\
 * \quad \text{and} \quad \bar{\alpha} &= \alpha \psi^{-6}
 * \f}
 *
 * are a set of nonlinear elliptic equations that the spacetime metric in
 * general relativity must satisfy at all times. For an introduction see e.g.
 * \cite BaumgarteShapiro, in particular Box 3.3 which is largely mirrored here.
 * We solve the XCTS equations for the conformal factor \f$\psi\f$, the product
 * of lapse times conformal factor \f$\alpha\psi\f$ and the shift vector
 * \f$\beta^j\f$. The remaining quantities in the equations, i.e. the conformal
 * metric \f$\bar{\gamma}_{ij}\f$, the trace of the extrinsic curvature \f$K\f$,
 * their respective time derivatives \f$\bar{u}_{ij}\f$ and \f$\partial_t K\f$,
 * the energy density \f$\rho\f$, the stress-energy trace \f$S\f$ and the
 * momentum density \f$S^i\f$, are freely specifyable fields that define the
 * physical scenario at hand. Of particular importance is the conformal metric,
 * which defines the background geometry, the covariant derivative
 * \f$\bar{D}\f$, the Ricci scalar \f$\bar{R}\f$ and the longitudinal operator
 *
 * \f{equation}
 * \left(\bar{L}\beta\right)^{ij} = \bar{D}^i\beta^j + \bar{D}^j\beta^i
 * - \frac{2}{3}\bar{\gamma}^{ij}\bar{D}_k\beta^k
 * \text{.}
 * \f}
 *
 * Note that the XCTS equations are essentially two Poisson equations and one
 * Elasticity equation with nonlinear sources on a curved geometry. In this
 * analogy, the longitudinal operator plays the role of the elastic constitutive
 * relation that connects the symmetric "shift strain"
 * \f$\bar{D}_{(i}\beta_{j)}\f$ with the "stress" \f$(\bar{L}\beta)^{ij}\f$ of
 * which we take the divergence in the momentum constraint. This particular
 * constitutive relation is equivalent to an isotropic and homogeneous material
 * with bulk modulus \f$K=0\f$ (not to be confused with the extrinsic curvature
 * trace \f$K\f$ in this context) and shear modulus \f$\mu=1\f$ (see
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`).
 *
 * Once the XCTS equations are solved we can construct the spatial metric and
 * extrinsic curvature as
 *
 * \f{align}
 * \gamma_{ij} &= \psi^4\bar{\gamma}_{ij} \\
 * K_{ij} &= \psi^{-2}\bar{A}_{ij} + \frac{1}{3}\gamma_{ij} K
 * \f}
 *
 * from which we can compose the full spacetime metric.
 */
namespace Xcts {}
