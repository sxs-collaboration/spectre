// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
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

struct CellLightCrossingTimeCompute : Tags::CellLightCrossingTime<DataVector>,
                                           db::ComputeTag {
  using base = Tags::CellLightCrossingTime<DataVector>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
    evolution::dg::subcell::Tags::Mesh<3>,
    evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
    gr::Tags::Lapse<DataVector>,
    gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
    gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>;

  static void function(
    gsl::not_null<return_type*> cell_light_crossing_time_,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coordinates,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric){
    cell_light_crossing_time(cell_light_crossing_time_, mesh,
        inertial_coordinates, lapse, shift, inv_spatial_metric);
  }
};


}  // namespace Particles::MonteCarlo
