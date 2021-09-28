// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Outflow.hpp"

#include <algorithm>
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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
Outflow::Outflow(CkMigrateMessage* const msg) : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow::get_clone() const {
  return std::make_unique<Outflow>(*this);
}

void Outflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

std::optional<std::string> Outflow::dg_outflow(
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
  // The characteristic speeds are bounded by \pm \alpha - \beta_n, and
  // saturate that bound, so there is no need to check the hydro-dependent
  // characteristic speeds.
  min_speed = std::min(min(-get(lapse) - get(normal_dot_shift)),
                       min(get(lapse) - get(normal_dot_shift)));
  if (min_speed < 0.0) {
    return {MakeString{} << "Outflow boundary condition violated. Speed: "
                         << min_speed << "\nn_i: "
                         << outward_directed_normal_covector << "\n"};
  }
  return std::nullopt;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
