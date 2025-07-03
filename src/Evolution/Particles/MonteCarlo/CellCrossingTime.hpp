// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;

template <size_t Dim>
class Mesh;
/// \endcond

namespace Particles::MonteCarlo {

void cell_light_crossing_time(
    gsl::not_null<Scalar<DataVector>*> cell_light_crossing_time,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coordinates,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric);

// Estimate of the time needed to cross a cell of the subcell mesh.
// Currently fills the answer with 1.0 when the DG grid is active;
// this will need to be modified to calculate the cell-crossing time
// after projection of the metric variables on the Dg grid, or to first
// calculate the cell-crossing time on the Dg grid (assuming that the cell
// size is that of the FD grid), then project the result.
struct CellLightCrossingTimeCompute : Tags::CellLightCrossingTime<DataVector>,
                                           db::ComputeTag {
  using base = Tags::CellLightCrossingTime<DataVector>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
                 evolution::dg::subcell::Tags::ActiveGrid>;

  static void function(
      gsl::not_null<return_type*> cell_light_crossing_time_,
      const Mesh<3>& mesh,
      const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coordinates,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const evolution::dg::subcell::ActiveGrid& active_grid) {
    if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
      cell_light_crossing_time(cell_light_crossing_time_, mesh,
                               inertial_coordinates, lapse, shift,
                               inv_spatial_metric);
    } else {
      cell_light_crossing_time_->get() =
          DataVector{mesh.number_of_grid_points(), 1.0};
    }
  }
};


}  // namespace Particles::MonteCarlo
