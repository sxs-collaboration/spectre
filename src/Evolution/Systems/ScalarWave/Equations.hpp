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
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;
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
 * \brief Computes the penalty flux for the ScalarWave system
 *
 * The penalty flux is given by:
 *
 * \f{align}
 * G(\Psi) &= F(\Psi) = 0 \\
 * G(\Pi) &= (n_i F^i(\Pi))_{\mathrm{int}} +
 *     \frac{1}{2} p \left(v^{-}_{\mathrm{int}} - v^{+}_{\mathrm{ext}}\right) \\
 * G(\Phi_i) &= (n_i F(\Phi_i))_{\mathrm{int}} - \frac{1}{2} p \left(
 *      (n_i v^{-})_{\mathrm{int}} + (n_i v^{+})_{\mathrm{ext}}\right)
 * \f}
 *
 * where \f$G\f$ is the interface normal dotted with numerical flux, the
 * first terms on the RHS for \f$G(\Pi)\f$ and \f$G(\Phi_i)\f$ are the fluxes
 * dotted with the interface normal (computed in
 * ScalarWave::ComputeNormalDotFluxes), and \f$v^{\pm}\f$ are outgoing and
 * incoming characteristic fields of the system (see characteristic_fields() for
 * their definition). The penalty factor is chosen to be \f$p=1\f$.
 */
template <size_t Dim>
struct PenaltyFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
 private:
  struct NormalTimesVPlus : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "NormalTimesVPlus"; }
  };
  struct NormalTimesVMinus : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "NormalTimesVMinus"; }
  };

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the penalty flux for a scalar wave system. It requires no "
      "options."};
  static std::string name() noexcept { return "Penalty"; }

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  using variables_tags = tmpl::list<Pi, Phi<Dim>, Psi>;

  using package_field_tags =
      tmpl::list<::Tags::NormalDotFlux<Pi>, ::Tags::NormalDotFlux<Phi<Dim>>,
                 Tags::VPlus, Tags::VMinus, NormalTimesVPlus,
                 NormalTimesVMinus>;
  using package_extra_tags = tmpl::list<>;

  using argument_tags =
      tmpl::list<::Tags::NormalDotFlux<Pi>, ::Tags::NormalDotFlux<Phi<Dim>>,
                 Tags::VPlus, Tags::VMinus,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  void package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_n_dot_flux_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_n_dot_flux_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_v_plus,
      gsl::not_null<Scalar<DataVector>*> packaged_v_minus,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_n_times_v_plus,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_n_times_v_minus,
      const Scalar<DataVector>& normal_dot_flux_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_phi,
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> pi_normal_dot_numerical_flux,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_normal_dot_numerical_flux,
      gsl::not_null<Scalar<DataVector>*> psi_normal_dot_numerical_flux,
      const Scalar<DataVector>& normal_dot_flux_pi_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_phi_interior,
      const Scalar<DataVector>& v_plus_interior,
      const Scalar<DataVector>& v_minus_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_v_plus_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_v_minus_interior,
      const Scalar<DataVector>& minus_normal_dot_flux_pi_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_dot_flux_phi_exterior,
      const Scalar<DataVector>& v_plus_exterior,
      const Scalar<DataVector>& v_minus_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_times_v_plus_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_times_v_minus_exterior) const noexcept;
};

/*!
 * \ingroup NumericalFluxesGroup
 * \brief Computes the upwind flux
 *
 * The upwind flux is given by:
 * \f{align}
 * G(\Psi) =& 0 \\
 * G(\Pi) =& \frac{1}{2}\left(F(\Pi)_{\mathrm{int}} + F(\Pi)_{\mathrm{ext}}
 *                    + \Pi_{\mathrm{int}} - \Pi_{\mathrm{ext}}
 *                    + \gamma_2\left(\Psi_{\mathrm{ext}} -
 *                                    \Psi_{\mathrm{int}}\right)\right) \\
 * G(\Phi_i) =& \frac{1}{2} \left(F(\Phi_i)_{\mathrm{int}}
 *                   + F(\Phi_i)_{\mathrm{ext}}
 *                   + (n_i)_{\mathrm{int}} F(\Pi)_{\mathrm{int}}
 *                   - (n_i)_{\mathrm{ext}} F(\Pi)_{\mathrm{ext}}
 *                   - \gamma_2\left((n_i)_{\mathrm{int}}\Psi_{\mathrm{int}}
 *                     - (n_i)_{\mathrm{ext}}\Psi_{\mathrm{ext}}\right)\right)
 * \f}
 * where \f$\gamma_2\f$ is the constraint damping parameter,  \f$G\f$ is
 * the normal dotted with the numerical flux and \f$F\f$ is the normal
 * dotted with the flux, which is computed in
 * ScalarWave::ComputeNormalDotFluxes
 */
template <size_t Dim>
struct UpwindFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
 private:
  struct NormalTimesFluxPi : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  };
  struct NormalTimesGamma2Psi : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  };
  struct Gamma2Psi : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the upwind flux for a scalar wave system. It requires no "
      "options."};
  static std::string name() noexcept { return "Upwind"; }

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  using variables_tags = tmpl::list<Pi, Phi<Dim>, Psi>;

  using package_field_tags =
      tmpl::list<::Tags::NormalDotFlux<Pi>, ::Tags::NormalDotFlux<Phi<Dim>>, Pi,
                 NormalTimesFluxPi, Gamma2Psi, NormalTimesGamma2Psi>;
  using package_extra_tags = tmpl::list<>;

  using argument_tags =
      tmpl::list<::Tags::NormalDotFlux<Pi>, ::Tags::NormalDotFlux<Phi<Dim>>, Pi,
                 Psi, Tags::ConstraintGamma2,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  void package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_n_dot_flux_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_n_dot_flux_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_n_times_flux_pi,
      gsl::not_null<Scalar<DataVector>*> packaged_gamma2_psi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_times_gamma2_psi,
      const Scalar<DataVector>& normal_dot_flux_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_phi,
      const Scalar<DataVector>& pi, const Scalar<DataVector>& psi,
      const Scalar<DataVector>& constraint_gamma2,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

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
      const Scalar<DataVector>& gamma2_psi_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_gamma2_psi_interior,
      const Scalar<DataVector>& minus_normal_dot_flux_pi_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_dot_flux_phi_exterior,
      const Scalar<DataVector>& pi_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_flux_pi_exterior,
      const Scalar<DataVector>& gamma2_psi_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_times_gamma2_psi_exterior) const noexcept;
};

/// Compute the maximum magnitude of the characteristic speeds.
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  SPECTRE_ALWAYS_INLINE static constexpr double apply() noexcept { return 1.0; }
};
}  // namespace ScalarWave
