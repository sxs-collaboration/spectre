// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
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
  using argument_tags =
      tmpl::list<Pi, Phi<Dim>,
                 Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>;
  static void apply(gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
                    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
                        phi_normal_dot_flux,
                    gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
                    const Scalar<DataVector>& pi,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        interface_unit_normal) noexcept;
};

/*!
 * \ingroup NumericalFluxesGroup
 * \brief Computes the upwind flux
 *
 * The upwind flux is given by:
 * \f{align}
 * F^*(\Psi) =& 0 \\
 * F^*(\Pi) =& \frac{1}{2}\left(F(\Pi)_{\mathrm{int}} + F(\Pi)_{\mathrm{ext}}
 *                    + \Pi_{\mathrm{int}} - \Pi_{\mathrm{ext}}\right) \\
 * F^*(\Phi_i) =& \frac{1}{2} \left(F(\Phi_i)_{\mathrm{int}}
 *                   + F(\Phi_i)_{\mathrm{ext}}
 *                   + (n_i)_{\mathrm{int}} F(\Pi)_{\mathrm{int}}
 *                   - (n_i)_{\mathrm{ext}} F(\Pi)_{\mathrm{ext}}\right)
 * \f}
 * where \f$F^*\f$ is the normal dotted with the numerical flux and \f$F\f$ is
 * the normal dotted with the flux, which is computed in
 * ScalarWave::ComputeNormalDotFluxes
 */
template <size_t Dim>
struct UpwindFlux {
 private:
  struct NormalTimesFluxPi {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "NormalTimesFluxPi"; }
  };

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the upwind flux for a scalar wave system. It requires no "
      "options."};

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  // This is the data needed to compute the numerical flux.
  // `dg::SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_tags =
      tmpl::list<Tags::NormalDotFlux<Pi>, Tags::NormalDotFlux<Phi<Dim>>, Pi,
                 NormalTimesFluxPi>;

  // These tags on the interface of the element are passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags =
      tmpl::list<Tags::NormalDotFlux<Pi>, Tags::NormalDotFlux<Phi<Dim>>, Pi,
                 Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // Following the not-null pointer to packaged_data, this function expects as
  // arguments the databox types of the `argument_tags`.
  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const Scalar<DataVector>& normal_dot_flux_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_phi,
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // The arguments are first the system::variables_tag::tags_list wrapped in
  // Tags::NormalDotNumericalFLux as not-null pointers to write the results
  // into, then the package_tags on the interior side of the mortar followed by
  // the package_tags on the exterior side.
  void operator()(
      gsl::not_null<Scalar<DataVector>*> pi_normal_dot_numerical_flux,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_normal_dot_numerical_flux,
      gsl::not_null<Scalar<DataVector>*> psi_normal_dot_numerical_flux,
      const Scalar<DataVector>& normal_dot_flux_pi_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_phi_interior,
      const Scalar<DataVector>& pi_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_flux_pi_interior,
      const Scalar<DataVector>& minus_normal_dot_flux_pi_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_dot_flux_phi_exterior,
      const Scalar<DataVector>& pi_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_flux_pi_exterior) const noexcept;
};

/// Compute the maximum magnitude of the characteristic speeds.
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  SPECTRE_ALWAYS_INLINE static constexpr double apply() noexcept { return 1.0; }
};
}  // namespace ScalarWave
