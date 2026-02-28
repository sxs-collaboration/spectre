// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class ComplexDataVector;
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the self force of a scalar charge on a Kerr
 * background.
 *
 * \see ScalarSelfForce::FirstOrderSystem
 */
namespace ScalarSelfForce {}

/// Tags for the ScalarSelfForce system.
namespace ScalarSelfForce::Tags {

/*!
 * \brief The complex m-mode field $\Psi_m$.
 *
 * Defined by the m-mode decomposition of the complex scalar field $\Phi$
 * [Eq. (2.6) in \cite Osburn:2022bby ]:
 *
 * \begin{equation}
 * \Phi(t,r,\theta,\phi) = \frac{1}{r} \sum_{m=-\infty}^{\infty}
 *   \Psi_m(r,\theta) e^{im\Delta\phi(r)} e^{im(\phi - \Omega t)}
 * \end{equation}
 *
 * where $\Delta\phi(r) = \frac{a}{r_\plus - r_\minus}
 * \ln(\frac{r-r_\plus}{r-r_\minus})$.
 */
struct MMode : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The factor multiplying the angular derivative in the principal part of
 * the equations.
 *
 * This is the factor $\alpha$ that defines the principal part of the equations
 * and allows to write it in first-order flux form given by
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i F^i = S_m
 * \end{equation}
 * with the flux
 * \begin{equation}
 * F^i = \{\partial_{r_\star}, \alpha \partial_{\cos\theta}\} \Psi_m
 * \text{.}
 * \end{equation}
 * This factor is set by the analytic data class (see
 * `ScalarSelfForce::AnalyticData::CircularOrbit`).
 */
struct Alpha : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The factor multiplying the non-derivative terms in the equations.
 *
 * This is the factor $\beta$ in the general form of the equations
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i F^i = S_m
 * \text{.}
 * \end{equation}
 * This factor is set by the analytic data class (see
 * `ScalarSelfForce::AnalyticData::CircularOrbit`).
 */
struct Beta : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The factor multiplying the first-derivative terms in the equations.
 *
 * This is the factor $\gamma_i$ in the general form of the equations
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i F^i = S_m
 * \text{.}
 * \end{equation}
 * This factor is set by the analytic data class (see
 * `ScalarSelfForce::AnalyticData::CircularOrbit`).
 */
struct Gamma : db::SimpleTag {
  using type = tnsr::i<ComplexDataVector, 2>;
};

/*!
 * \brief A flag indicating that we are solving for the regularized field in
 * this element.
 *
 * In elements at and around the scalar point charge we use the effective source
 * approach to split the m-mode field into a singular and a regular field
 * [Eq. (3.6) in \cite Osburn:2022bby ]:
 * \begin{equation}
 * \Psi_m = \Psi_m^\mathcal{P} + \Psi_m^\mathcal{R}
 * \text{.}
 * \end{equation}
 * In these elements, we solve for $\Psi_m^\mathcal{R}$ rather than $\Psi_m$.
 */
struct FieldIsRegularized : db::SimpleTag {
  using type = bool;
};

/*!
 * \brief The singular field $\Psi_m^\mathcal{P}$.
 *
 * Only defined where `FieldIsRegularized` is true.
 */
struct SingularField : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The Boyer-Lindquist radius $r$.
 *
 * Computed numerically from the tortoise radial coordinate $r_\star$
 * [see Eq. (2.12) in \cite Osburn:2022bby ].
 */
struct BoyerLindquistRadius : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The hyperboloidal boost function $H(r_*)$.
 */
struct BoostFunction : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The radial derivative of the boost function, $H'(r_*)$.
 */
struct BoostFunctionDeriv : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

}  // namespace ScalarSelfForce::Tags
