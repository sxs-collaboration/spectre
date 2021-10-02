// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

// IWYU pragma: no_forward_declare hydro::Tags::SpatialVelocity
// IWYU pragma: no_forward_declare hydro::Tags::LorentzFactor
// IWYU pragma: no_forward_declare hydro::Tags::RestMassDensity
// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {
/*!
 *\ingroup VariableFixingGroup
 * \brief Limit the maximum Lorentz factor to LorentzFactorCap in regions where
 * the density is below MaxDensityCutoff.
 *
 * The Lorentz factor is set to LorentzFactorCap and the spatial velocity is
 * adjusted according to:
 * \f{align}
 * v^i_{(\textrm{new})} = \sqrt{\left(1 - \frac{1}{W_{(\textrm{new})}^2}\right)
 * \left(1 - \frac{1}{W_{(\textrm{old})}^2}\right)^{-1}} v^i_{(\textrm{old})}
 * \f}
 */
class LimitLorentzFactor {
 public:
  /// Do not apply the Lorentz factor cap above this density
  struct MaxDensityCutoff {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Do not apply the Lorentz factor cap above this density"};
  };
  /// Largest Lorentz factor allowed. If a larger one is found, normalize
  /// velocity to have the Lorentz factor be this value.
  struct LorentzFactorCap {
    using type = double;
    static type lower_bound() { return 1.0; }
    static constexpr Options::String help = {"Largest Lorentz factor allowed."};
  };

  using options = tmpl::list<MaxDensityCutoff, LorentzFactorCap>;
  static constexpr Options::String help = {
      "Limit the maximum Lorentz factor to LorentzFactorCap in regions where "
      "the\n"
      "density is below MaxDensityCutoff. The Lorentz factor is set to\n"
      "LorentzFactorCap and the spatial velocity is adjusted accordingly."};

  LimitLorentzFactor(double max_density_cutoff, double lorentz_factor_cap);

  LimitLorentzFactor() = default;
  LimitLorentzFactor(const LimitLorentzFactor& /*rhs*/) = default;
  LimitLorentzFactor& operator=(const LimitLorentzFactor& /*rhs*/) = default;
  LimitLorentzFactor(LimitLorentzFactor&& /*rhs*/) = default;
  LimitLorentzFactor& operator=(LimitLorentzFactor&& /*rhs*/) = default;
  ~LimitLorentzFactor() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  using return_tags = tmpl::list<hydro::Tags::LorentzFactor<DataVector>,
                                 hydro::Tags::SpatialVelocity<DataVector, 3>>;

  using argument_tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>>;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      const Scalar<DataVector>& rest_mass_density) const;

 private:
  friend bool operator==(const LimitLorentzFactor& lhs,
                         const LimitLorentzFactor& rhs);

  double max_density_cuttoff_;
  double lorentz_factor_cap_;
};

bool operator!=(const LimitLorentzFactor& lhs, const LimitLorentzFactor& rhs);
}  // namespace VariableFixing
