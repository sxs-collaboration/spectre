// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TagsDeclarations.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace RelativisticEuler {
namespace Valencia {

/*!
 * \brief Fix conservative variables using the method proposed in F. Foucart's
 * PhD thesis (Cornell)
 *
 * Fix the conservative variables at each grid point according to the following
 * recipe:
 *
 * - If \f$D < D_\text{cutoff}\f$, set \f$D = D_\text{min}\f$
 * - If \f$\tau < 0\f$, set \f$\tau = 0\f$
 * - If \f$\tilde S^2 > \tilde S_\text{max}^2 = \tilde\tau(\tilde\tau +
 *   2\tilde D)\f$, rescale \f$\tilde S_i\f$ by the factor
 *
 * \f{align*}
 * f = \sqrt{\frac{(1 - \epsilon)\tilde S_\text{max}^2}{\tilde S^2 +
 * 10^{-16}\tilde D^2}} < 1 \f}
 *
 * (if \f$ f\geq 1\f$, do not fix \f$\tilde S_i\f$),
 * where \f$D_\text{min}\f$, \f$D_\text{cutoff}\f$ and \f$\epsilon\f$
 * are small and positive input parameters, typically chosen in consistency with
 * some variable fixing applied to the primitives variables
 * (e.g. VariableFixing::FixToAtmosphere). Note that, in order to fix
 * \f$\tilde S_i\f$, \f$\epsilon\f$ is chosen so that \f$1 - \epsilon\f$
 * (the relevant quantity used in the equations)
 * is sufficiently close (but smaller) than 1.
 */
template <size_t Dim>
class FixConservatives {
 public:
  /// The minimum value of the rest mass density times the Lorentz factor,
  /// \f$D\f$
  struct MinimumValueOfD {
    using type = double;
    static constexpr Options::String help = {
        "Minimum value of rest-mass density times Lorentz factor"};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The cutoff below which \f$D\f$ is set to `MinimumValueOfD`
  struct CutoffD {
    using type = double;
    static constexpr Options::String help = {
        "Cutoff below which D is set to MinimumValueOfD"};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The safety factor to fix \f$\tilde S_i\f$
  struct SafetyFactorForS {
    using type = double;
    static constexpr Options::String help = {
        "Safety factor for momentum density bound."};
    static type lower_bound() noexcept {
      return std::numeric_limits<double>::epsilon();
    }
    static type upper_bound() noexcept { return 1.0; }
  };

  using options = tmpl::list<MinimumValueOfD, CutoffD, SafetyFactorForS>;
  static constexpr Options::String help = {
      "Variable fixing used in Foucart's thesis."};

  FixConservatives() = default;
  FixConservatives(const FixConservatives& /*rhs*/) = default;
  FixConservatives& operator=(const FixConservatives& /*rhs*/) = default;
  FixConservatives(FixConservatives&& /*rhs*/) noexcept = default;
  FixConservatives& operator=(FixConservatives&& /*rhs*/) noexcept = default;
  ~FixConservatives() = default;

  FixConservatives(double minimum_rest_mass_density_times_lorentz_factor,
                   double rest_mass_density_times_lorentz_factor_cutoff,
                   double safety_factor_for_momentum_density,
                   const Options::Context& context = {});

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  using return_tags =
      tmpl::list<RelativisticEuler::Valencia::Tags::TildeD,
                 RelativisticEuler::Valencia::Tags::TildeTau,
                 RelativisticEuler::Valencia::Tags::TildeS<Dim>>;

  using argument_tags = tmpl::list<gr::Tags::InverseSpatialMetric<Dim>,
                                   gr::Tags::SqrtDetSpatialMetric<>>;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> tilde_s,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric) const noexcept;

 private:
  template <size_t SpatialDim>
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const FixConservatives<SpatialDim>& lhs,
      const FixConservatives<SpatialDim>& rhs) noexcept;

  double minimum_rest_mass_density_times_lorentz_factor_ =
      std::numeric_limits<double>::signaling_NaN();
  double rest_mass_density_times_lorentz_factor_cutoff_ =
      std::numeric_limits<double>::signaling_NaN();
  double one_minus_safety_factor_for_momentum_density_ =
      std::numeric_limits<double>::signaling_NaN();
};

template <size_t Dim>
bool operator!=(const FixConservatives<Dim>& lhs,
                const FixConservatives<Dim>& rhs) noexcept;
}  // namespace Valencia
}  // namespace RelativisticEuler
