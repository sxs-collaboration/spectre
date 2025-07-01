// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class ComplexDataVector;
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the gravitational self-force of a gravitating
 * body in a Kerr background.
 *
 * \see GrSelfForce::FirstOrderSystem
 */
namespace GrSelfForce {}

/// Tags for the GrSelfForce system.
namespace GrSelfForce::Tags {

/*!
 * \brief The complex m-mode field $(\Psi_m)_{ab}$.
 *
 * Defined by the m-mode decomposition of the metric perturbation $h_{ab}$:
 * \begin{equation}
 * \bar{h}_{ab}(t,r,\theta,\phi) = \sum_{m=-\infty}^{\infty}
 *   \bar{h}^m_{ab}(r,\theta) e^{im(\phi - \Omega t)}
 * \end{equation}
 * and further decomposition with convenient prefactors:
 * \begin{align*}
 * &(\Psi_m)_{tt} = r \bar{h}^m_{tt} \\
 * &(\Psi_m)_{tr} = \frac{\Delta}{r} \bar{h}^m_{tr} \\
 * &(\Psi_m)_{t\theta} = \bar{h}^m_{t\theta} \\
 * &(\Psi_m)_{t\phi} = \frac{1}{\sin\theta} \bar{h}^m_{t\phi} \\
 * &(\Psi_m)_{rr} = \frac{\Delta^2}{r^3} \bar{h}^m_{rr} \\
 * &(\Psi_m)_{r\theta} = \frac{\Delta}{r^2} \bar{h}^m_{r\theta} \\
 * &(\Psi_m)_{r\phi} = \frac{\Delta}{r^2\sin\theta} \bar{h}^m_{r\phi} \\
 * &(\Psi_m)_{\theta\theta} = \frac{1}{r} \bar{h}^m_{\theta\theta} \\
 * &(\Psi_m)_{\theta\phi} = \frac{1}{r\sin\theta} \bar{h}^m_{\theta\phi} \\
 * &(\Psi_m)_{\phi\phi} = \frac{1}{r\sin^2\theta} \bar{h}^m_{\phi\phi} \\
 * \end{align*}
 */
struct MMode : db::SimpleTag {
  using type = tnsr::aa<ComplexDataVector, 3>;
};

/*!
 * \brief The factor multiplying the angular derivative in the principal part of
 * the equations.
 *
 * This is the factor $\alpha$ that defines the principal part of the equations
 * and allows to write it in first-order flux form (see
 * `GrSelfForce::FirstOrderSystem`). This factor is set by the analytic data
 * class (see `ScalarSelfForce::AnalyticData::CircularOrbit`).
 */
struct Alpha : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The factors $\beta_{ab}^{cd}$ multiplying the non-derivative terms in
 * the equations.
 */
struct Beta : db::SimpleTag {
  using type = tnsr::aaBB<ComplexDataVector, 3>;
};

/*!
 * \brief The factors $\gamma_{{r_\star}ab}^{cd}$ multiplying the
 * $r_\star$-derivative terms in the equations.
 */
struct GammaRstar : db::SimpleTag {
  using type = tnsr::aaBB<ComplexDataVector, 3>;
};

/*!
 * \brief The factors $\gamma_{\theta ab}^{cd}$ multiplying the
 * $\theta$-derivative terms in the equations.
 */
struct GammaTheta : db::SimpleTag {
  using type = tnsr::aaBB<ComplexDataVector, 3>;
};

/*!
 * \brief A flag indicating that we are solving for the regularized field in
 * this element.
 *
 * In elements at and around the point mass we use the effective source
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
  using type = tnsr::aa<ComplexDataVector, 3>;
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

}  // namespace GrSelfForce::Tags
