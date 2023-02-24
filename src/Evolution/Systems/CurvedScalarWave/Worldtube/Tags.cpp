// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <optional>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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

template struct CenteredFaceCoordinatesCompute<3>;
}  // namespace CurvedScalarWave::Worldtube::Tags
