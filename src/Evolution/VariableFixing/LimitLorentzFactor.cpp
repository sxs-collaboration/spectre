// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/LimitLorentzFactor.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {
LimitLorentzFactor::LimitLorentzFactor(const double max_density_cutoff,
                                       const double lorentz_factor_cap) noexcept
    : max_density_cuttoff_(max_density_cutoff),
      lorentz_factor_cap_(lorentz_factor_cap) {}

void LimitLorentzFactor::pup(PUP::er& p) noexcept {
  p | max_density_cuttoff_;
  p | lorentz_factor_cap_;
}

void LimitLorentzFactor::operator()(
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const Scalar<DataVector>& rest_mass_density) const noexcept {
  constexpr size_t dim = 3;
  const size_t number_of_grid_points = get(rest_mass_density).size();
  for (size_t s = 0; s < number_of_grid_points; ++s) {
    if (get(*lorentz_factor)[s] > lorentz_factor_cap_ and
        get(rest_mass_density)[s] < max_density_cuttoff_) {
      const double velocity_renorm_factor =
          sqrt((1. - 1. / square(lorentz_factor_cap_)) /
               (1. - 1. / square(get(*lorentz_factor)[s])));

      get(*lorentz_factor)[s] = lorentz_factor_cap_;
      for (size_t d = 0; d < dim; ++d) {
        spatial_velocity->get(d)[s] *= velocity_renorm_factor;
      }
    }
  }
}

bool operator==(const LimitLorentzFactor& lhs,
                const LimitLorentzFactor& rhs) noexcept {
  return lhs.max_density_cuttoff_ == rhs.max_density_cuttoff_ and
         lhs.lorentz_factor_cap_ == rhs.lorentz_factor_cap_;
}

bool operator!=(const LimitLorentzFactor& lhs,
                const LimitLorentzFactor& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace VariableFixing
