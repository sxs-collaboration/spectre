// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <optional>
#include <type_traits>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Tags {

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <size_t Dim, typename Frame, bool Centered>
void FaceCoordinatesCompute<Dim, Frame, Centered>::function(
    const gsl::not_null<std::optional<tnsr::I<DataVector, Dim, Frame>>*> result,
    const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
    const tnsr::I<DataVector, Dim, Frame>& coords, const Mesh<Dim>& mesh) {
  const auto direction = excision_sphere.abutting_direction(element.id());
  if (direction.has_value()) {
    ASSERT(
        mesh.quadrature(direction.value().dimension()) ==
            Spectral::Quadrature::GaussLobatto,
        "Expected GaussLobatto quadrature. Other quadratures are disabled "
        "because interpolating the coordinates incurs an unnecessary error.");
    const size_t grid_size =
        mesh.slice_away(direction->dimension()).number_of_grid_points();
    if (result->has_value()) {
      destructive_resize_components(make_not_null(&(result->value())),
                                    grid_size);
    } else {
      result->emplace(grid_size);
    }
    data_on_slice(make_not_null(&(result->value())), coords, mesh.extents(),
                  direction.value().dimension(),
                  index_to_slice_at(mesh.extents(), direction.value()));
    if constexpr (Centered) {
      if constexpr (not std::is_same_v<Frame, ::Frame::Grid>) {
        ERROR("Should be grid frame");
      }
      for (size_t i = 0; i < Dim; ++i) {
        result->value().get(i) -= excision_sphere.center().get(i);
      }
    }
  } else {
    result->reset();
  }
}

template <size_t Dim, typename Frame, bool Centered>
void FaceCoordinatesCompute<Dim, Frame, Centered>::function(
    const gsl::not_null<
        std::optional<tnsr::I<DataVector, Dim, ::Frame::Inertial>>*>
        result,
    const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
    const tnsr::I<DataVector, Dim, ::Frame::Inertial>& inertial_coords,
    const Mesh<Dim>& mesh, const tnsr::I<double, Dim>& particle_position) {
  if constexpr (not(Centered and std::is_same_v<Frame, ::Frame::Inertial>)) {
    ERROR("Should be centered in inertial frame");
  }
  const auto direction = excision_sphere.abutting_direction(element.id());
  if (direction.has_value()) {
    ASSERT(
        mesh.quadrature(direction.value().dimension()) ==
            Spectral::Quadrature::GaussLobatto,
        "Expected GaussLobatto quadrature. Other quadratures are disabled "
        "because interpolating the coordinates incurs an unnecessary error.");
    const size_t grid_size =
        mesh.slice_away(direction->dimension()).number_of_grid_points();
    if (result->has_value()) {
      destructive_resize_components(make_not_null(&(result->value())),
                                    grid_size);
    } else {
      result->emplace(grid_size);
    }
    data_on_slice(make_not_null(&(result->value())), inertial_coords,
                  mesh.extents(), direction.value().dimension(),
                  index_to_slice_at(mesh.extents(), direction.value()));
    for (size_t i = 0; i < Dim; ++i) {
      result->value().get(i) -= particle_position.get(i);
    }
  } else {
    result->reset();
  }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

template <size_t Dim>
void PunctureFieldCompute<Dim>::function(
    const gsl::not_null<return_type*> result,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        inertial_face_coords,
    const ::ExcisionSphere<Dim>& excision_sphere, const double time,
    const size_t expansion_order) {
  if (inertial_face_coords.has_value()) {
    if (not result->has_value()) {
      result->emplace(get<0>(inertial_face_coords.value()).size());
    }
    puncture_field(
        make_not_null(&(result->value())), inertial_face_coords.value(), time,
        get(magnitude(excision_sphere.center())), 1., expansion_order);
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

template struct InertialParticlePositionCompute<3>;
template struct PunctureFieldCompute<3>;

template struct FaceCoordinatesCompute<3, Frame::Grid, true>;
template struct FaceCoordinatesCompute<3, Frame::Grid, false>;
template struct FaceCoordinatesCompute<3, Frame::Inertial, true>;
template struct FaceCoordinatesCompute<3, Frame::Inertial, false>;

}  // namespace CurvedScalarWave::Worldtube::Tags
