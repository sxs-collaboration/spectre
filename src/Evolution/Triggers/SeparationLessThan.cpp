// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Triggers/SeparationLessThan.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"

namespace Triggers {
SeparationLessThan::SeparationLessThan(const double separation)
    : separation_(separation) {}

bool SeparationLessThan::operator()(
    const double time, const ::Domain<3>& domain, const Element<3>& element,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const tnsr::I<double, 3, Frame::Grid>& object_center_a,
    const tnsr::I<double, 3, Frame::Grid>& object_center_b) const {
  const ElementId<3>& element_id = element.id();
  const size_t block_id = element_id.block_id();

  const Block<3>& block = domain.blocks()[block_id];

  tnsr::I<double, 3, Frame::Inertial> inertial_object_center_a{};
  tnsr::I<double, 3, Frame::Inertial> inertial_object_center_b{};

  if (block.has_distorted_frame()) {
    tnsr::I<double, 3, Frame::Distorted> distorted_object_center_a{};
    tnsr::I<double, 3, Frame::Distorted> distorted_object_center_b{};

    for (size_t i = 0; i < 3; i++) {
      distorted_object_center_a.get(i) = object_center_a.get(i);
      distorted_object_center_b.get(i) = object_center_b.get(i);
    }

    const auto& dist_to_inertial_map =
        block.moving_mesh_distorted_to_inertial_map();

    inertial_object_center_a = dist_to_inertial_map(distorted_object_center_a,
                                                    time, functions_of_time);
    inertial_object_center_b = dist_to_inertial_map(distorted_object_center_b,
                                                    time, functions_of_time);
  } else {
    const auto& grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map();

    inertial_object_center_a =
        grid_to_inertial_map(object_center_a, time, functions_of_time);
    inertial_object_center_b =
        grid_to_inertial_map(object_center_b, time, functions_of_time);
  }

  const tnsr::I<double, 3, Frame::Inertial> position_difference =
      tenex::evaluate<ti::I>(inertial_object_center_a(ti::I) -
                             inertial_object_center_b(ti::I));

  const double calculated_separation =
      sqrt(square(get<0>(position_difference)) +
           square(get<1>(position_difference)) +
           square(get<2>(position_difference)));

  return calculated_separation <= separation_;
}

void SeparationLessThan::pup(PUP::er& p) { p | separation_; }

PUP::able::PUP_ID SeparationLessThan::my_PUP_ID = 0;  // NOLINT
}  // namespace Triggers
