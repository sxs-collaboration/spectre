// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Freezing.hpp"

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
namespace helpers {
template <typename T>
void set_bc_corr_zero_when_char_speed_is_positive(
    const gsl::not_null<T*> dt_v_corr,
    const DataVector& char_speed_u) noexcept {
  for (DataVector& component : *dt_v_corr) {
    for (size_t i = 0; i < component.size(); ++i) {
      if (char_speed_u[i] > 0.) {
        component[i] = 0.;
      }
    }
  }
}
}  // namespace helpers

template <size_t Dim>
FreezingBjorhus<Dim>::FreezingBjorhus(CkMigrateMessage* const msg) noexcept
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
FreezingBjorhus<Dim>::get_clone() const noexcept {
  return std::make_unique<FreezingBjorhus>(*this);
}

template <size_t Dim>
void FreezingBjorhus<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
std::optional<std::string> FreezingBjorhus<Dim>::dg_time_derivative(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        dt_phi_correction,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::AA<DataVector, Dim, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& dt_phi) const noexcept {
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataVector>,
                        ::Tags::TempII<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempiaa<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<1, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<2, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<3, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempiaa<1, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<4, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<5, Dim, Frame::Inertial, DataVector>>>
      local_buffer(get_size(get<0>(normal_covector)), 0.);

  auto& temp_scalar = get<::Tags::TempScalar<0, DataVector>>(local_buffer);
  auto& inverse_spatial_metric =
      get<::Tags::TempII<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_psi =
      get<::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_zero =
      get<::Tags::Tempiaa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_minus =
      get<::Tags::Tempaa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_plus =
      get<::Tags::Tempaa<2, Dim, Frame::Inertial, DataVector>>(local_buffer);

  auto& bc_dt_v_psi =
      get<::Tags::Tempaa<3, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_zero =
      get<::Tags::Tempiaa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_plus =
      get<::Tags::Tempaa<4, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_minus =
      get<::Tags::Tempaa<5, Dim, Frame::Inertial, DataVector>>(local_buffer);

  // Calculate char speeds
  typename Tags::CharacteristicSpeeds<Dim, Frame::Inertial>::type char_speeds;
  characteristic_speeds(make_not_null(&char_speeds), gamma1, lapse, shift,
                        normal_covector);

  // Account for moving mesh: char speeds -> cher speeds - n_i v^i_g
  if (face_mesh_velocity.has_value()) {
    get(temp_scalar) = get(dot_product(normal_covector, *face_mesh_velocity));
    for (size_t a = 0; a < 4; ++a) {
      char_speeds.at(a) -= get(temp_scalar);
    }
  }

  // Initialize with zeros
  std::fill(dt_spacetime_metric_correction->begin(),
            dt_spacetime_metric_correction->end(), 0.);
  std::fill(dt_pi_correction->begin(), dt_pi_correction->end(), 0.);
  std::fill(dt_phi_correction->begin(), dt_phi_correction->end(), 0.);

  // Calculate char projections of RHS
  get(temp_scalar) = 1. / (get(lapse) * get(lapse));
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inverse_spatial_metric.get(i, j) =
          inverse_spacetime_metric.get(1 + i, 1 + j) +
          (shift.get(i) * shift.get(j) * get(temp_scalar));
    }
  }
  const auto dt_char_fields =
      characteristic_fields(gamma2, inverse_spatial_metric, dt_spacetime_metric,
                            dt_pi, dt_phi, normal_covector);
  char_projected_rhs_dt_v_psi =
      get<Tags::VSpacetimeMetric<Dim, Frame::Inertial>>(dt_char_fields);
  char_projected_rhs_dt_v_zero =
      get<Tags::VZero<Dim, Frame::Inertial>>(dt_char_fields);
  char_projected_rhs_dt_v_plus =
      get<Tags::VPlus<Dim, Frame::Inertial>>(dt_char_fields);
  char_projected_rhs_dt_v_minus =
      get<Tags::VMinus<Dim, Frame::Inertial>>(dt_char_fields);

  // unconditionally freeze char fields, where needed
  for (size_t a = 0; a <= Dim; ++a) {
    for (size_t b = a; b <= Dim; ++b) {
      bc_dt_v_psi.get(a, b) = -char_projected_rhs_dt_v_psi.get(a, b);
      bc_dt_v_plus.get(a, b) = -char_projected_rhs_dt_v_plus.get(a, b);
      bc_dt_v_minus.get(a, b) = -char_projected_rhs_dt_v_minus.get(a, b);
      for (size_t i = 0; i < Dim; ++i) {
        bc_dt_v_zero.get(i, a, b) = -char_projected_rhs_dt_v_zero.get(i, a, b);
      }
    }
  }

  // Only add corrections at grid points where the char speeds are negative
  helpers::set_bc_corr_zero_when_char_speed_is_positive(
      make_not_null(&bc_dt_v_psi), char_speeds[0]);
  helpers::set_bc_corr_zero_when_char_speed_is_positive(
      make_not_null(&bc_dt_v_zero), char_speeds[1]);
  helpers::set_bc_corr_zero_when_char_speed_is_positive(
      make_not_null(&bc_dt_v_plus), char_speeds[2]);
  helpers::set_bc_corr_zero_when_char_speed_is_positive(
      make_not_null(&bc_dt_v_minus), char_speeds[3]);

  // The boundary conditions here are imposed using Bjorhus' method
  // see, e.g. Bjorhus.hpp for details
  auto dt_evolved_vars = evolved_fields_from_characteristic_fields(
      gamma2, bc_dt_v_psi, bc_dt_v_zero, bc_dt_v_plus, bc_dt_v_minus,
      normal_covector);

  *dt_pi_correction = get<Tags::Pi<Dim, Frame::Inertial>>(dt_evolved_vars);
  *dt_phi_correction = get<Tags::Phi<Dim, Frame::Inertial>>(dt_evolved_vars);
  *dt_spacetime_metric_correction =
      get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
          dt_evolved_vars);

  if (face_mesh_velocity.has_value()) {
    get(temp_scalar) = get(dot_product(normal_covector, *face_mesh_velocity));
    // we use 1e-10 instead of 0 below to allow for purely tangentially
    // moving grids, eg a rotating sphere, with some leeway for
    // floating-point errors.
    if (max(get(temp_scalar)) > 1.e-10) {
      return {
          "We found the radial mesh velocity points in the direction "
          "of the outward normal, i.e. we possibly have an expanding "
          "domain. Its unclear if proper boundary conditions are "
          "imposed in this case."};
    }
  }

  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID FreezingBjorhus<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class FreezingBjorhus<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic::BoundaryConditions
