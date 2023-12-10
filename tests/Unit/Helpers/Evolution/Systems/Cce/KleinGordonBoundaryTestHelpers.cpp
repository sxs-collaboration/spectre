// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/Systems/Cce/KleinGordonBoundaryTestHelpers.hpp"

namespace Cce::TestHelpers {

void create_fake_time_varying_klein_gordon_data(
    const gsl::not_null<Scalar<ComplexModalVector>*> kg_psi_modal,
    const gsl::not_null<Scalar<ComplexModalVector>*> kg_pi_modal,
    const gsl::not_null<Scalar<DataVector>*> kg_psi_nodal,
    const gsl::not_null<Scalar<DataVector>*> kg_pi_nodal,
    const double extraction_radius, const double amplitude,
    const double frequency, const double time, const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  tnsr::I<DataVector, 3> collocation_points{number_of_angular_points};

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    get<0>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        sin(collocation_point.theta) * cos(collocation_point.phi);
    get<1>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        sin(collocation_point.theta) * sin(collocation_point.phi);
    get<2>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        cos(collocation_point.theta);
  }

  const DataVector r = sqrt(square(get<0>(collocation_points)) +
                            square(get<1>(collocation_points)) +
                            square(get<2>(collocation_points)));
  const DataVector dr_dt = r / (1.0 + amplitude * sin(frequency * time)) *
                           amplitude * frequency * cos(frequency * time);

  // Nodal data
  get(*kg_psi_nodal) = sin(r - time);
  get(*kg_pi_nodal) = cos(r - time) * (dr_dt - 1.0);

  // Transform to modal data
  *kg_psi_modal =
      TestHelpers::tensor_to_goldberg_coefficients(*kg_psi_nodal, l_max);
  *kg_pi_modal =
      TestHelpers::tensor_to_goldberg_coefficients(*kg_pi_nodal, l_max);
}

}  // namespace Cce::TestHelpers
