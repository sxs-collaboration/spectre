// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/LaneEmdenGravitationalField.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler::Sources {

LaneEmdenGravitationalField::LaneEmdenGravitationalField(
    const double central_mass_density, const double polytropic_constant)
    : central_mass_density_(central_mass_density),
      polytropic_constant_(polytropic_constant) {}

LaneEmdenGravitationalField::LaneEmdenGravitationalField(CkMigrateMessage* msg)
    : Source{msg} {}

void LaneEmdenGravitationalField::pup(PUP::er& p) { Source::pup(p); }

auto LaneEmdenGravitationalField::get_clone() const
    -> std::unique_ptr<Source<3>> {
  return std::make_unique<LaneEmdenGravitationalField>(*this);
}

void LaneEmdenGravitationalField::operator()(
    const gsl::not_null<Scalar<DataVector>*> /*source_mass_density_cons*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& /*energy_density*/,
    const tnsr::I<DataVector, 3>& /*velocity*/,
    const Scalar<DataVector>& /*pressure*/,
    const Scalar<DataVector>& /*specific_internal_energy*/,
    const EquationsOfState::EquationOfState<false, 2>& /*eos*/,
    const tnsr::I<DataVector, 3>& coords, const double /*time*/) const {
  const auto gravitational_field = this->gravitational_field(coords);
  for (size_t i = 0; i < 3; ++i) {
    source_momentum_density->get(i) +=
        get(mass_density_cons) * gravitational_field.get(i);
  }
  get(*source_energy_density) +=
      get(dot_product(momentum_density, gravitational_field));
}

tnsr::I<DataVector, 3> LaneEmdenGravitationalField::gravitational_field(
    const tnsr::I<DataVector, 3>& x) const {
  // Compute alpha for polytrope n==1, units G==1
  const double alpha = sqrt(0.5 * polytropic_constant_ / M_PI);
  const double outer_radius = alpha * M_PI;
  const double mass_scale = 4.0 * M_PI * cube(alpha) * central_mass_density_;
  // Add tiny offset to avoid divisons by zero
  const DataVector radius = get(magnitude(x)) + 1.e-30 * outer_radius;

  auto enclosed_mass =
      make_with_value<DataVector>(get_size(radius), mass_scale);
  for (size_t s = 0; s < get_size(radius); ++s) {
    if (get_element(radius, s) < outer_radius) {
      const double xi = get_element(radius, s) / alpha;
      get_element(enclosed_mass, s) *= sin(xi) - xi * cos(xi);
    } else {
      get_element(enclosed_mass, s) *= M_PI;
    }
  }

  auto gravitational_field_result = x;
  for (size_t i = 0; i < 3; ++i) {
    gravitational_field_result.get(i) *= -enclosed_mass / cube(radius);
  }
  return gravitational_field_result;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID LaneEmdenGravitationalField::my_PUP_ID = 0;
}  // namespace NewtonianEuler::Sources
