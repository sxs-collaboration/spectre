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
#include "Domain/ExcisionSphere.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

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
      set_number_of_grid_points(make_not_null(&(result->value())), grid_size);
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
    const Mesh<Dim>& mesh,
    const std::array<tnsr::I<double, Dim>, 2>& particle_position_velocity) {
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
      set_number_of_grid_points(make_not_null(&(result->value())), grid_size);
    } else {
      result->emplace(grid_size);
    }
    data_on_slice(make_not_null(&(result->value())), inertial_coords,
                  mesh.extents(), direction.value().dimension(),
                  index_to_slice_at(mesh.extents(), direction.value()));
    for (size_t i = 0; i < Dim; ++i) {
      result->value().get(i) -= particle_position_velocity.at(0).get(i);
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
        inertial_face_coords_centered,
    const std::array<tnsr::I<double, Dim, ::Frame::Inertial>, 2>&
        particle_position_velocity,
    const tnsr::I<double, Dim>& particle_acceleration, const double charge,
    const size_t expansion_order) {
  if (inertial_face_coords_centered.has_value()) {
    if (not result->has_value()) {
      result->emplace(get<0>(inertial_face_coords_centered.value()).size());
    }
    puncture_field(make_not_null(&(result->value())),
                   inertial_face_coords_centered.value(),
                   particle_position_velocity[0], particle_position_velocity[1],
                   particle_acceleration, 1., expansion_order);
    result->value() *= charge;
  } else {
    result->reset();
  }
}

template <size_t Dim>
void ParticlePositionVelocityCompute<Dim>::function(
    gsl::not_null<std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>*>
        position_and_velocity,
    const ::ExcisionSphere<Dim>& excision_sphere, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  ASSERT(excision_sphere.is_time_dependent(),
         "The worldtube simulation requires time-dependent maps.");
  const auto& grid_to_inertial_map =
      excision_sphere.moving_mesh_grid_to_inertial_map();
  auto mapped_tuple = grid_to_inertial_map.coords_frame_velocity_jacobians(
      excision_sphere.center(), time, functions_of_time);
  (*position_and_velocity)[0] = std::move(std::get<0>(mapped_tuple));
  (*position_and_velocity)[1] = std::move(std::get<3>(mapped_tuple));
}

template <size_t Dim>
void EvolvedParticlePositionVelocityCompute<Dim>::function(
    gsl::not_null<std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>*>
        position_velocity,
    const tnsr::I<DataVector, Dim>& evolved_position,
    const tnsr::I<DataVector, Dim>& evolved_velocity) {
  for (size_t i = 0; i < Dim; ++i) {
    (*position_velocity)[0].get(i) = evolved_position.get(i)[0];
    (*position_velocity)[1].get(i) = evolved_velocity.get(i)[0];
  }
}

template <size_t Dim>
void GeodesicAccelerationCompute<Dim>::function(
    gsl::not_null<tnsr::I<double, Dim, Frame::Inertial>*> acceleration,
    const std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>&
        position_velocity,
    const gr::Solutions::KerrSchild& background_spacetime) {
  const auto christoffel = get<
      gr::Tags::SpacetimeChristoffelSecondKind<double, Dim, Frame::Inertial>>(
      background_spacetime.variables(
          position_velocity.at(0), 0.,
          tmpl::list<gr::Tags::SpacetimeChristoffelSecondKind<
              double, Dim, Frame::Inertial>>{}));
  gr::geodesic_acceleration(acceleration, position_velocity.at(1), christoffel);
}

template struct ParticlePositionVelocityCompute<3>;
template struct EvolvedParticlePositionVelocityCompute<3>;
template struct GeodesicAccelerationCompute<3>;
template struct PunctureFieldCompute<3>;

template struct FaceCoordinatesCompute<3, Frame::Grid, true>;
template struct FaceCoordinatesCompute<3, Frame::Grid, false>;
template struct FaceCoordinatesCompute<3, Frame::Inertial, true>;
template struct FaceCoordinatesCompute<3, Frame::Inertial, false>;

}  // namespace CurvedScalarWave::Worldtube::Tags
