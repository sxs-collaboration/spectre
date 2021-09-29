// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // For Tags::Normalized
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
/// \ingroup ComputationalDomainGroup
/// \brief %Tags for the domain.
namespace Tags {
/// The Inertial coordinates, the inverse Jacobian from the Grid to the Inertial
/// frame, the Jacobian from the Grid to the Inertial frame, and the Inertial
/// mesh velocity.
///
/// The type is a `std::optional`, which, when is not valid, signals that the
/// mesh is not moving. Thus,
/// `coordinates_velocity_and_jacobian.has_value()` can be used to check
/// if the mesh is moving.
template <size_t Dim>
struct CoordinatesMeshVelocityAndJacobians : db::SimpleTag {
  using type = std::optional<std::tuple<
      tnsr::I<DataVector, Dim, Frame::Inertial>,
      ::InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
      ::Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
      tnsr::I<DataVector, Dim, Frame::Inertial>>>;
};

/// Computes the Inertial coordinates, the inverse Jacobian from the Grid to the
/// Inertial frame, the Jacobian from the Grid to the Inertial frame, and the
/// Inertial mesh velocity.
template <typename MapTagGridToInertial>
struct CoordinatesMeshVelocityAndJacobiansCompute
    : CoordinatesMeshVelocityAndJacobians<MapTagGridToInertial::dim>,
      db::ComputeTag {
  static constexpr size_t dim = MapTagGridToInertial::dim;
  using base = CoordinatesMeshVelocityAndJacobians<dim>;

  using return_type = typename base::type;

  static void function(
      const gsl::not_null<return_type*> result,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, dim>&
          grid_to_inertial_map,
      const tnsr::I<DataVector, dim, Frame::Grid>& source_coords,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) noexcept {
    // Use identity to signal time-independent
    if (not grid_to_inertial_map.is_identity()) {
      *result = grid_to_inertial_map.coords_frame_velocity_jacobians(
          source_coords, time, functions_of_time);
    } else {
      *result = std::nullopt;
    }
  }

  using argument_tags =
      tmpl::list<MapTagGridToInertial, Tags::Coordinates<dim, Frame::Grid>,
                 ::Tags::Time, Tags::FunctionsOfTime>;
};

/// Computes the Inertial coordinates from
/// `CoordinatesVelocityAndJacobians`
template <size_t Dim>
struct InertialFromGridCoordinatesCompute
    : Tags::Coordinates<Dim, Frame::Inertial>,
      db::ComputeTag {
  using base = Tags::Coordinates<Dim, Frame::Inertial>;
  using return_type = typename base::type;

  static void function(
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> target_coords,
      const tnsr::I<DataVector, Dim, Frame::Grid>& source_coords,
      const std::optional<std::tuple<
          tnsr::I<DataVector, Dim, Frame::Inertial>,
          ::InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
          ::Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
          tnsr::I<DataVector, Dim, Frame::Inertial>>>&
          grid_to_inertial_quantities) noexcept;

  using argument_tags = tmpl::list<Tags::Coordinates<Dim, Frame::Grid>,
                                   CoordinatesMeshVelocityAndJacobians<Dim>>;
};

/// Computes the Logical to Inertial inverse Jacobian from
/// `CoordinatesVelocityAndJacobians`
template <size_t Dim>
struct ElementToInertialInverseJacobian
    : Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Inertial>,
      db::ComputeTag {
  using base =
      Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Inertial>;
  using return_type = typename base::type;

  static void function(
      gsl::not_null<::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                      Frame::Inertial>*>
          inv_jac_logical_to_inertial,
      const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                              Frame::Grid>& inv_jac_logical_to_grid,
      const std::optional<std::tuple<
          tnsr::I<DataVector, Dim, Frame::Inertial>,
          ::InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
          ::Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
          tnsr::I<DataVector, Dim, Frame::Inertial>>>&
          grid_to_inertial_quantities) noexcept;

  using argument_tags =
      tmpl::list<Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Grid>,
                 CoordinatesMeshVelocityAndJacobians<Dim>>;
};

/// The mesh velocity
///
/// The type is a `std::optional`, which when it is not set indicates that the
/// mesh is not moving.
template <size_t Dim, typename Frame = ::Frame::Inertial>
struct MeshVelocity : db::SimpleTag {
  using type = std::optional<tnsr::I<DataVector, Dim, Frame>>;
};

/// Computes the Inertial mesh velocity from `CoordinatesVelocityAndJacobians`
///
/// The type is a `std::optional`, which when it is not set indicates that the
/// mesh is not moving.
template <size_t Dim>
struct InertialMeshVelocityCompute : MeshVelocity<Dim, Frame::Inertial>,
                                     db::ComputeTag {
  using base = MeshVelocity<Dim, Frame::Inertial>;
  using return_type = typename base::type;

  static void function(
      gsl::not_null<return_type*> mesh_velocity,
      const std::optional<std::tuple<
          tnsr::I<DataVector, Dim, Frame::Inertial>,
          ::InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
          ::Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
          tnsr::I<DataVector, Dim, Frame::Inertial>>>&
          grid_to_inertial_quantities) noexcept;

  using argument_tags = tmpl::list<CoordinatesMeshVelocityAndJacobians<Dim>>;
};

/// The divergence of the mesh velocity
struct DivMeshVelocity : db::SimpleTag {
  using type = std::optional<Scalar<DataVector>>;
  static std::string name() noexcept { return "div(MeshVelocity)"; }
};
}  // namespace Tags
}  // namespace domain
