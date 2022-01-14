// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryConditions/Outflow.hpp"

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"

namespace Burgers::BoundaryConditions {
Outflow::Outflow(CkMigrateMessage* const msg) : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow::get_clone() const {
  return std::make_unique<Outflow>(*this);
}

void Outflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

std::optional<std::string> Outflow::dg_outflow(
    const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 1, Frame::Inertial>&
        outward_directed_normal_covector,
    const Scalar<DataVector>& u) {
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

void Outflow::fd_outflow(
    const gsl::not_null<Scalar<DataVector>*> u, const Direction<1>& direction,
    const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 1, Frame::Inertial>&
        outward_directed_normal_covector,
    const Scalar<DataVector>& u_interior, const Mesh<1>& subcell_mesh) {
  // The outflow condition here simply uses the outermost values on
  // cell-centered FD grid points to compute face values on the external
  // boundary. This is equivalent to adopting the piecewise constant FD
  // reconstruction for FD cells at the external boundaries.
  //
  const double u_val_at_boundary = get(
      u_interior)[direction.side() == Side::Upper ? subcell_mesh.extents(0) - 1
                                                  : 0];

  double min_char_speed = std::numeric_limits<double>::signaling_NaN();
  if (face_mesh_velocity.has_value()) {
    min_char_speed = min(get<0>(outward_directed_normal_covector) *
                         (u_val_at_boundary - get<0>(*face_mesh_velocity)));
  } else {
    min_char_speed =
        min(get<0>(outward_directed_normal_covector) * u_val_at_boundary);
  }
  if (min_char_speed < 0.0) {
    ERROR("Outflow boundary condition (subcell) violated with speed U ingoing:"
          << min_char_speed << "\nU: " << u_val_at_boundary
          << "\nn_i: " << outward_directed_normal_covector << "\n");
  } else {
    // Once the outflow condition has been checked, we fill the ghost data with
    // the boundary values. This does not mirror the data across the boundary
    // and so is quite low-order.
    //
    // The reason that we need this step is to prevent floating point exceptions
    // being raised while computing the subcell time derivative because of NaN
    // or uninitialized values in ghost data.

    get(*u) = u_val_at_boundary;
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow::my_PUP_ID = 0;
}  // namespace Burgers::BoundaryConditions
