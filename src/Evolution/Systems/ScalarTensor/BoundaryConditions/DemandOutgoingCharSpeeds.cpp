// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"

namespace ScalarTensor::BoundaryConditions {
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DemandOutgoingCharSpeeds::get_clone() const {
  return std::make_unique<DemandOutgoingCharSpeeds>(*this);
}

void DemandOutgoingCharSpeeds::pup(PUP::er& p) { BoundaryCondition::pup(p); }

DemandOutgoingCharSpeeds::DemandOutgoingCharSpeeds(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::optional<std::string>
DemandOutgoingCharSpeeds::dg_demand_outgoing_char_speeds(
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3>& outward_directed_normal_covector,
    const tnsr::I<DataVector, 3>& /*outward_directed_normal_vector*/,
    const Scalar<DataVector>& gh_gamma1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3>& shift, const Scalar<DataVector>& csw_gamma1) {
  tnsr::a<DataVector, 3, Frame::Inertial> csw_char_speeds{lapse.size()};
  ::CurvedScalarWave::characteristic_speeds(make_not_null(&csw_char_speeds),
                                            csw_gamma1, lapse, shift,
                                            outward_directed_normal_covector);

  std::array<DataVector, 4> gh_char_speeds = ::gh::characteristic_speeds(
      gh_gamma1, lapse, shift, outward_directed_normal_covector,
      face_mesh_velocity);

  if (face_mesh_velocity.has_value()) {
    const auto face_speed =
        dot_product(outward_directed_normal_covector, *face_mesh_velocity);
    for (auto& char_speed : csw_char_speeds) {
      char_speed -= get(face_speed);
    }
    for (auto& char_speed : gh_char_speeds) {
      char_speed -= get(face_speed);
    }
  }
  for (size_t i = 0; i < csw_char_speeds.size(); ++i) {
    if (min(csw_char_speeds[i]) < 0.) {
      return MakeString{}
             << "Detected negative characteristic speed at boundary with "
                "outgoing char speeds boundary conditions specified. The "
                "speed is "
             << min(csw_char_speeds[i]) << " for index " << i
             << ". To see which characteristic field this corresponds to, "
                "check the function `characteristic_speeds` in "
                "Evolution/Systems/CurvedScalarWave/Characteristics.hpp.";
    }

    if (min(gsl::at(gh_char_speeds, i)) < 0.) {
      return MakeString{}
             << "Detected negative characteristic speed at boundary with "
                "outgoing char speeds boundary conditions specified. The "
                "speed is "
             << min(gsl::at(gh_char_speeds, i)) << " for index " << i
             << ". To see which characteristic field this corresponds to, "
                "check the function `characteristic_speeds` in "
                "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp.";
    }
  }
  return std::nullopt;  // LCOV_EXCL_LINE
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DemandOutgoingCharSpeeds::my_PUP_ID = 0;
}  // namespace ScalarTensor::BoundaryConditions
