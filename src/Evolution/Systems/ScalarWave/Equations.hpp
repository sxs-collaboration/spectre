// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

template <typename>
class Variables;

class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

namespace Tags {
template <typename, typename, typename>
struct deriv;
template <typename>
struct dt;
template <typename>
struct NormalDotFlux;
template <typename>
struct NormalDotNumericalFlux;
template <typename>
struct Variables;
}  // namespace Tags

namespace ScalarWave {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;
}  // namespace ScalarWave

namespace ScalarWave {
/*!
 * \brief Compute the time derivative of the evolved variables of the
 * first-order scalar wave system.
 *
 * The evolution equations for the first-order scalar wave system are given by:
 * \f{align}
 * \partial_t\psi = & -\pi \\
 * \partial_t\Phi_i = & -\partial_i \pi \\
 * \partial_t\pi = & - \delta^{ij}\partial_i\Phi_j
 * \f}
 *
 * where \f$\psi\f$ is the scalar field, \f$\pi=-\partial_t\psi\f$ is the
 * conjugate momentum to \f$\psi\f$, and \f$\Phi_i=\partial_i\psi\f$ is an
 * auxiliary variable.
 */
template <size_t Dim>
struct ComputeDuDt {
  using return_tags = typelist<Tags::dt<Pi>, Tags::dt<Phi<Dim>>, Tags::dt<Psi>>;

  using argument_tags =
      tmpl::list<Pi, Tags::deriv<Pi, tmpl::size_t<Dim>, Frame::Inertial>,
                 Tags::deriv<Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> dt_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
      gsl::not_null<Scalar<DataVector>*> dt_psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) noexcept;
};

/*!
 * \brief Compute normal component of flux on a boundary.
 *
 * \f{align}
 * F(\Psi) =& 0 \\
 * F(\Pi) =& n^i \Phi_i \\
 * F(\Phi_i) =& n_i \Pi
 * \f}
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
  using return_tags =
      tmpl::list<Tags::NormalDotFlux<Pi>, Tags::NormalDotFlux<Phi<Dim>>,
                 Tags::NormalDotFlux<Psi>>;
  using argument_tags = tmpl::list<Pi, Phi<Dim>>;
  static void apply(gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
                    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
                        phi_normal_dot_flux,
                    gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
                    const Scalar<DataVector>& pi,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        interface_unit_normal) noexcept;
};
}  // namespace ScalarWave
