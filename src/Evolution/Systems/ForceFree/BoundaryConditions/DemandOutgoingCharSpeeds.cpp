// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::BoundaryConditions {
DemandOutgoingCharSpeeds::DemandOutgoingCharSpeeds(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DemandOutgoingCharSpeeds::get_clone() const {
  return std::make_unique<DemandOutgoingCharSpeeds>(*this);
}

void DemandOutgoingCharSpeeds::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID DemandOutgoingCharSpeeds::my_PUP_ID = 0;

std::optional<std::string>
DemandOutgoingCharSpeeds::dg_demand_outgoing_char_speeds(
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
    /*outward_directed_normal_vector*/,

    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse) {
  double min_speed = std::numeric_limits<double>::signaling_NaN();

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get(lapse).size()};

  auto& normal_dot_shift = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_shift),
              outward_directed_normal_covector, shift);

  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    get(normal_dot_shift) += get(normal_dot_mesh_velocity);
  }

  // The characteristic speeds are bounded by \pm \alpha - \beta^i n_i,
  // therefore minimum is given as `-\alpha - \beta^i n_i`.
  min_speed = min(-get(lapse) - get(normal_dot_shift));

  if (min_speed < 0.0) {
    return {MakeString{}
            << "DemandOutgoingCharSpeeds boundary condition violated. Speed: "
            << min_speed << "\nn_i: " << outward_directed_normal_covector
            << "\n"};
  }

  return std::nullopt;
}

}  // namespace ForceFree::BoundaryConditions
