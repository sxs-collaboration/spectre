// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryConditions/Outflow.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Utilities/MakeString.hpp"

namespace Burgers::BoundaryConditions {
Outflow::Outflow(CkMigrateMessage* const msg) noexcept
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow::get_clone() const noexcept {
  return std::make_unique<Outflow>(*this);
}

void Outflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

std::optional<std::string> Outflow::dg_outflow(
    const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 1, Frame::Inertial>&
        outward_directed_normal_covector,
    const Scalar<DataVector>& u) noexcept {
  double min_speed = std::numeric_limits<double>::signaling_NaN();
  if (face_mesh_velocity.has_value()) {
    min_speed = min(get<0>(outward_directed_normal_covector) *
                    (get(u) - get<0>(*face_mesh_velocity)));
  } else {
    min_speed = min(get<0>(outward_directed_normal_covector) * get(u));
  }
  if (min_speed < 0.0) {
    return {MakeString{}
            << "Outflow boundary condition violated with speed U ingoing: "
            << min_speed << "\nU: " << u
            << "\nn_i: " << outward_directed_normal_covector << "\n"};
  }
  return std::nullopt;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow::my_PUP_ID = 0;
}  // namespace Burgers::BoundaryConditions
