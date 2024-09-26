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
#include "Domain/AreaElement.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace CurvedScalarWave::Worldtube::OptionTags {
SelfForceOptions::SelfForceOptions() = default;
SelfForceOptions::SelfForceOptions(const double mass_in,
                                   const size_t iterations_in,
                                   const double turn_on_time_in,
                                   const double turn_on_interval_in)
    : mass(mass_in),
      iterations(iterations_in),
      turn_on_time(turn_on_time_in),
      turn_on_interval(turn_on_interval_in) {}

void SelfForceOptions::pup(PUP::er& p) {
  p | mass;
  p | iterations;
  p | turn_on_time;
  p | turn_on_interval;
}
}  // namespace CurvedScalarWave::Worldtube::OptionTags

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

template <size_t Dim>
void BackgroundQuantitiesCompute<Dim>::function(
    gsl::not_null<return_type*> result,
    const std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>&
        position_velocity,
    const gr::Solutions::KerrSchild& background_spacetime) {
  auto kerr_schild_quantities = background_spacetime.variables(
      position_velocity.at(0), 0.,
      tmpl::list<gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                 gr::Tags::SpatialMetric<double, Dim>,
                 gr::Tags::InverseSpatialMetric<double, Dim>,
                 gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>{});
  auto metric = gr::spacetime_metric(
      get<gr::Tags::Lapse<double>>(kerr_schild_quantities),
      get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities),
      get<gr::Tags::SpatialMetric<double, Dim>>(kerr_schild_quantities));
  auto inverse_metric = gr::inverse_spacetime_metric(
      get<gr::Tags::Lapse<double>>(kerr_schild_quantities),
      get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities),
      get<gr::Tags::InverseSpatialMetric<double, Dim>>(kerr_schild_quantities));

  const auto& velocity = position_velocity.at(1);
  double temp = metric.get(0, 0);
  for (size_t i = 0; i < Dim; ++i) {
    temp += 2. * metric.get(i + 1, 0) * velocity.get(i);
    for (size_t j = 0; j < Dim; ++j) {
      temp += metric.get(i + 1, j + 1) * velocity.get(i) * velocity.get(j);
    }
  }
  auto& christoffel =
      get<gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>(
          kerr_schild_quantities);
  auto& trace_christoffel =
      get<gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>>(*result);
  tenex::evaluate<ti::C>(
      make_not_null(&trace_christoffel),
      inverse_metric(ti::A, ti::B) * christoffel(ti::C, ti::a, ti::b));
  get(get<TimeDilationFactor>(*result)) = sqrt(-1. / temp);
  get<gr::Tags::SpacetimeMetric<double, Dim>>(*result) = std::move(metric);
  get<gr::Tags::InverseSpacetimeMetric<double, Dim>>(*result) =
      std::move(inverse_metric);
  get<gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>(*result) =
      std::move(christoffel);
}

void FaceQuantitiesCompute::function(
    gsl::not_null<return_type*> result, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
    const tnsr::I<DataVector, Dim>& shift, const Scalar<DataVector>& lapse,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
    const Mesh<Dim>& mesh) {
  const auto direction = excision_sphere.abutting_direction(element.id());
  if (not direction.has_value()) {
    result->reset();
    return;
  }
  const auto face_mesh = mesh.slice_away(direction->dimension());
  const size_t face_size = face_mesh.number_of_grid_points();
  const auto slice_index = index_to_slice_at(mesh.extents(), direction.value());
  const auto sliced_dim = direction.value().dimension();
  Variables<tags_to_slice_to_face> vars_on_face(face_size);
  // we use a templated lambda here. This could also be solved with a
  // non-owning variables but this has less overhead.
  const auto slice_to_face =
      [&vars_on_face, &mesh, &sliced_dim, &slice_index]<typename tag_to_slice>(
          const typename tag_to_slice::type& volume_field) ->
      typename tag_to_slice::type& {
        data_on_slice(make_not_null(&get<tag_to_slice>(vars_on_face)),
                      volume_field, mesh.extents(), sliced_dim, slice_index);
        return get<tag_to_slice>(vars_on_face);
      };
  const auto& face_psi =
      slice_to_face.operator()<CurvedScalarWave::Tags::Psi>(psi);
  const auto& face_pi =
      slice_to_face.operator()<CurvedScalarWave::Tags::Pi>(pi);
  const auto& face_phi =
      slice_to_face.operator()<CurvedScalarWave::Tags::Phi<Dim>>(phi);
  const auto& face_shift =
      slice_to_face.operator()<gr::Tags::Shift<DataVector, Dim>>(shift);
  const auto& face_lapse =
      slice_to_face.operator()<gr::Tags::Lapse<DataVector>>(lapse);
  const auto& face_inv_jacobian =
      slice_to_face.operator()<domain::Tags::InverseJacobian<
          Dim, Frame::ElementLogical, Frame::Inertial>>(inv_jacobian);

  if (not result->has_value()) {
    result->emplace(face_size);
  }
  get<CurvedScalarWave::Tags::Psi>(result->value()) = face_psi;
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(result->value())) =
      -get(face_lapse) * get(face_pi) + get(dot_product(face_shift, face_phi));
  euclidean_area_element(
      make_not_null(
          &get<gr::surfaces::Tags::AreaElement<DataVector>>(result->value())),
      face_inv_jacobian, direction.value());
}

template struct BackgroundQuantitiesCompute<3>;
template struct EvolvedParticlePositionVelocityCompute<3>;
template struct GeodesicAccelerationCompute<3>;
template struct ParticlePositionVelocityCompute<3>;
template struct PunctureFieldCompute<3>;

template struct FaceCoordinatesCompute<3, Frame::Grid, true>;
template struct FaceCoordinatesCompute<3, Frame::Grid, false>;
template struct FaceCoordinatesCompute<3, Frame::Inertial, true>;
template struct FaceCoordinatesCompute<3, Frame::Inertial, false>;

}  // namespace CurvedScalarWave::Worldtube::Tags
