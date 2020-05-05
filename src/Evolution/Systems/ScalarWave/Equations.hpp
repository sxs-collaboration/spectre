// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename>
class Variables;

class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

namespace Tags {
template <typename>
struct NormalDotFlux;
template <typename>
struct Normalized;
}  // namespace Tags

namespace ScalarWave {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;
}  // namespace ScalarWave

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
/*!
 * \brief Compute the time derivative of the evolved variables of the
 * first-order scalar wave system.
 *
 * The evolution equations for the first-order scalar wave system are given by:
 * \f{align}
 * \partial_t\Psi = & -\Pi \\
 * \partial_t\Phi_i = & -\partial_i \Pi +
 *                       \gamma_2 (\partial_i\Psi - \Phi_i) \\
 * \partial_t\Pi = & - \delta^{ij}\partial_i\Phi_j
 * \f}
 *
 * where \f$\Psi\f$ is the scalar field, \f$\Pi=-\partial_t\Psi\f$ is the
 * conjugate momentum to \f$\Psi\f$, \f$\Phi_i=\partial_i\Psi\f$ is an
 * auxiliary variable, and \f$\gamma_2\f$ is the constraint damping parameter.
 */
template <size_t Dim>
struct ComputeDuDt {
  template <template <class> class StepPrefix>
  using return_tags = tmpl::list<db::add_tag_prefix<StepPrefix, Pi>,
                                 db::add_tag_prefix<StepPrefix, Phi<Dim>>,
                                 db::add_tag_prefix<StepPrefix, Psi>>;

  using argument_tags =
      tmpl::list<::Tags::deriv<Psi, tmpl::size_t<Dim>, Frame::Inertial>, Pi,
                 ::Tags::deriv<Pi, tmpl::size_t<Dim>, Frame::Inertial>,
                 Phi<Dim>,
                 ::Tags::deriv<Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
                 Tags::ConstraintGamma2>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> dt_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
      gsl::not_null<Scalar<DataVector>*> dt_psi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,
      const Scalar<DataVector>& gamma2) noexcept;
};

/*!
 * \brief A relic of an old incorrect way of handling boundaries for
 * non-conservative systems.
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
  using argument_tags = tmpl::list<Pi>;
  static void apply(gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
                    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
                        phi_normal_dot_flux,
                    gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
                    const Scalar<DataVector>& pi) noexcept;
};

/// Compute the maximum magnitude of the characteristic speeds.
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  SPECTRE_ALWAYS_INLINE static constexpr double apply() noexcept { return 1.0; }
};
}  // namespace ScalarWave
