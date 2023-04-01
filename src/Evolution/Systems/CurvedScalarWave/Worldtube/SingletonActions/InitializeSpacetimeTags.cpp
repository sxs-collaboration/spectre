// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeSpacetimeTags.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Initialization {

void InitializeSpacetimeTags::apply(
    const gsl::not_null<tnsr::AA<double, Dim, Frame::Grid>*>
        inverse_spacetime_metric,
    const gsl::not_null<tnsr::A<double, Dim, Frame::Grid>*>
        trace_spacetime_christoffel,
    const ExcisionSphere<Dim>& excision_sphere) {
  const double M = 1.;
  const double orbit_radius = get(magnitude(excision_sphere.center()));
  *inverse_spacetime_metric = tnsr::AA<double, Dim, Frame::Grid>(0.);
  get<0, 0>(*inverse_spacetime_metric) = -1. - 2. * M / orbit_radius;
  get<1, 1>(*inverse_spacetime_metric) = 1. - 2. * M / orbit_radius;
  get<2, 2>(*inverse_spacetime_metric) =
      (-2. * M + square(orbit_radius) - orbit_radius) / square(orbit_radius);
  get<3, 3>(*inverse_spacetime_metric) = 1.;
  get<0, 1>(*inverse_spacetime_metric) = 2. * M / orbit_radius;
  get<2, 0>(*inverse_spacetime_metric) =
      (2. * M + orbit_radius) / (orbit_radius * sqrt(orbit_radius));
  get<2, 1>(*inverse_spacetime_metric) =
      -2. * M / (orbit_radius * sqrt(orbit_radius));

  get<0>(*trace_spacetime_christoffel) = -2. * M / square(orbit_radius);
  get<1>(*trace_spacetime_christoffel) =
      -(2. * M + orbit_radius - 2. * M * orbit_radius) / cube(orbit_radius);
  get<2>(*trace_spacetime_christoffel) =
      6. * M / (square(orbit_radius) * sqrt(orbit_radius));
  get<3>(*trace_spacetime_christoffel) = 0.;
}
}  // namespace CurvedScalarWave::Worldtube::Initialization
