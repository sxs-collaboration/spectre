// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
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

/*!
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
    static constexpr db::DataBoxString label = "NormalTimesFluxPi";
  };

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the upwind flux for a scalar wave system. It requires no "
      "options."};

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  using return_tags = tmpl::list<Tags::NormalDotNumericalFlux<Pi>,
                                 Tags::NormalDotNumericalFlux<Phi<Dim>>,
                                 Tags::NormalDotNumericalFlux<Psi>>;

  using package_tags =
      tmpl::list<Tags::NormalDotFlux<Pi>, Tags::NormalDotFlux<Phi<Dim>>, Pi,
                 NormalTimesFluxPi>;

  using slice_tags = tmpl::list<Pi>;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const Scalar<DataVector>& normal_dot_flux_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_phi,
      const Scalar<DataVector>& /*normal_dot_flux_psi*/,
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
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
