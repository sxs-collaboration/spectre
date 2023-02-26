// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <optional>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Tags {
template <size_t Dim>
void CenteredFaceCoordinatesCompute<Dim>::function(
    const gsl::not_null<std::optional<tnsr::I<DataVector, Dim, Frame::Grid>>*>
        result,
    const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
    const tnsr::I<DataVector, Dim, Frame::Grid>& grid_coords,
    const Mesh<Dim>& mesh) {
  const auto direction = excision_sphere.abutting_direction(element.id());
  if (direction.has_value()) {
    const size_t grid_size =
        mesh.slice_away(direction->dimension()).number_of_grid_points();
    if (result->has_value()) {
      destructive_resize_components(make_not_null(&(result->value())),
                                    grid_size);
    } else {
      result->emplace(grid_size);
    }
    evolution::dg::project_tensor_to_boundary(make_not_null(&(result->value())),
                                              grid_coords, mesh,
                                              direction.value());
    for (size_t i = 0; i < Dim; ++i) {
      result->value().get(i) -= excision_sphere.center().get(i);
    }

  } else {
    result->reset();
  }
}

template <size_t Dim>
void InertialParticlePositionCompute<Dim>::function(
    gsl::not_null<tnsr::I<double, Dim, Frame::Inertial>*> inertial_position,
    const ::ExcisionSphere<Dim>& excision_sphere, const double time) {
  const auto& grid_position = excision_sphere.center();
  const double orbital_radius = get(magnitude(grid_position));

  // assume circular orbit around black hole with mass 1
  const double angular_velocity = 1. / (sqrt(orbital_radius) * orbital_radius);
  const double angle = angular_velocity * time;
  inertial_position->get(0) =
      cos(angle) * grid_position.get(0) - sin(angle) * grid_position.get(1);
  inertial_position->get(1) =
      sin(angle) * grid_position.get(0) + cos(angle) * grid_position.get(1);
  inertial_position->get(2) = grid_position.get(2);
}

template struct CenteredFaceCoordinatesCompute<3>;
template struct InertialParticlePositionCompute<3>;
}  // namespace CurvedScalarWave::Worldtube::Tags
