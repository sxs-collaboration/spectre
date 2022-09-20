// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Time/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace evolution::dg::subcell::Tags {
/// The coordinates in a given frame.
template <size_t Dim, typename Frame>
struct Coordinates : db::SimpleTag {
  static std::string name() { return get_output(Frame{}) + "Coordinates"; }
  using type = tnsr::I<DataVector, Dim, Frame>;
};

/// The element logical coordinates on the subcell grid
template <size_t VolumeDim>
struct LogicalCoordinatesCompute
    : Coordinates<VolumeDim, Frame::ElementLogical>,
      db::ComputeTag {
  using base = Coordinates<VolumeDim, Frame::ElementLogical>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Mesh<VolumeDim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<return_type*>, const ::Mesh<VolumeDim>&)>(
      &logical_coordinates<VolumeDim>);
};

/// The inertial coordinates on the subcell grid
template <typename MapTagGridToInertial>
struct InertialCoordinatesCompute
    : Coordinates<MapTagGridToInertial::dim, Frame::Inertial>,
      db::ComputeTag {
  static constexpr size_t dim = MapTagGridToInertial::dim;
  using base = Coordinates<dim, Frame::Inertial>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<MapTagGridToInertial, Tags::Coordinates<dim, Frame::Grid>,
                 ::Tags::Time, ::domain::Tags::FunctionsOfTime>;
  static void function(
      const gsl::not_null<return_type*> inertial_coords,
      const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, dim>&
          grid_to_inertial_map,
      const tnsr::I<DataVector, dim, Frame::Grid>& grid_coords,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    *inertial_coords =
        grid_to_inertial_map(grid_coords, time, functions_of_time);
  }
};
}  // namespace evolution::dg::subcell::Tags
