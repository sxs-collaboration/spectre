// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
namespace helpers {
double min_characteristic_speed(const std::array<DataVector, 4>& char_speeds) {
  std::array<double, 4> min_speeds{{min(char_speeds[0]), min(char_speeds[1]),
                                    min(char_speeds[2]), min(char_speeds[3])}};
  return *std::min_element(min_speeds.begin(), min_speeds.end());
}
template <typename T>
void set_bc_corr_zero_when_char_speed_is_positive(
    const gsl::not_null<T*> dt_v_corr, const DataVector& char_speed_u) {
  for (DataVector& component : *dt_v_corr) {
    for (size_t i = 0; i < component.size(); ++i) {
      if (char_speed_u[i] > 0.) {
        component[i] = 0.;
      }
    }
  }
}
}  // namespace helpers

namespace detail {
ConstraintPreservingBjorhusType
convert_constraint_preserving_bjorhus_type_from_yaml(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if (type_read == "ConstraintPreserving") {
    return ConstraintPreservingBjorhusType::ConstraintPreserving;
  } else if (type_read == "ConstraintPreservingPhysical") {
    return ConstraintPreservingBjorhusType::ConstraintPreservingPhysical;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert input option to "
              "ConstraintPreservingBjorhusType::Type. Must "
              "be one of ConstraintPreserving or ConstraintPreservingPhysical");
}
}  // namespace detail

template <size_t Dim>
ConstraintPreservingBjorhus<Dim>::ConstraintPreservingBjorhus(
    const detail::ConstraintPreservingBjorhusType type)
    : type_(type) {}

template <size_t Dim>
ConstraintPreservingBjorhus<Dim>::ConstraintPreservingBjorhus(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConstraintPreservingBjorhus<Dim>::get_clone() const {
  return std::make_unique<ConstraintPreservingBjorhus>(*this);
}

template <size_t Dim>
void ConstraintPreservingBjorhus<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | type_;
}

template <size_t Dim>
std::optional<std::string> ConstraintPreservingBjorhus<Dim>::dg_time_derivative(
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
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::AA<DataVector, Dim, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::A<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::a<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_one_form,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
    const tnsr::a<DataVector, Dim, Frame::Inertial>& gauge_source,
    const tnsr::ab<DataVector, Dim, Frame::Inertial>&
        spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, Frame::Inertial>&
        logical_dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& logical_dt_pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& logical_dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::ijaa<DataVector, Dim, Frame::Inertial>& d_phi) const {
  TempBuffer<tmpl::list<
      ::Tags::TempI<0, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempiaa<1, Dim, Frame::Inertial, DataVector>,
      ::Tags::TempII<0, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempii<0, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempa<1, Dim, Frame::Inertial, DataVector>,
      ::Tags::TempA<1, Dim, Frame::Inertial, DataVector>,
      ::Tags::TempA<2, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>,
      ::Tags::TempAb<0, Dim, Frame::Inertial, DataVector>,
      ::Tags::TempAA<1, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<1, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempiaa<2, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<2, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<3, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempa<2, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempa<3, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<4, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempiaa<3, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<5, Dim, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<6, Dim, Frame::Inertial, DataVector>,
      // inertial time derivatives
      ::Tags::dt<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>,
      ::Tags::dt<Tags::Pi<Dim, Frame::Inertial>>,
      ::Tags::dt<Tags::Phi<Dim, Frame::Inertial>>>>
      local_buffer(get_size(get<0>(normal_covector)), 0.);

  tnsr::aa<DataVector, Dim, Frame::Inertial> dt_spacetime_metric;
  tnsr::aa<DataVector, Dim, Frame::Inertial> dt_pi;
  tnsr::iaa<DataVector, Dim, Frame::Inertial> dt_phi;
  if (face_mesh_velocity.has_value()) {
    for (size_t storage_index = 0; storage_index < dt_pi.size();
         ++storage_index) {
      dt_spacetime_metric[storage_index].set_data_ref(make_not_null(
          &get<::Tags::dt<
              gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>>(
              local_buffer)[storage_index]));
      dt_pi[storage_index].set_data_ref(
          make_not_null(&get<::Tags::dt<Tags::Pi<Dim, Frame::Inertial>>>(
              local_buffer)[storage_index]));
    }
    for (size_t storage_index = 0; storage_index < dt_phi.size();
         ++storage_index) {
      dt_phi[storage_index].set_data_ref(
          make_not_null(&get<::Tags::dt<Tags::Phi<Dim, Frame::Inertial>>>(
              local_buffer)[storage_index]));
    }
    // Compute inertial time derivative by subtracting mesh velocity from
    // logical time derivative.
    for (size_t a = 0; a < Dim + 1; ++a) {
      for (size_t b = a; b < Dim + 1; ++b) {
        dt_spacetime_metric.get(a, b) = logical_dt_spacetime_metric.get(a, b);
        dt_pi.get(a, b) = logical_dt_pi.get(a, b);
        for (size_t d = 0; d < Dim; ++d) {
          dt_spacetime_metric.get(a, b) -=
              face_mesh_velocity->get(d) * d_spacetime_metric.get(d, a, b);
          dt_pi.get(a, b) -= face_mesh_velocity->get(d) * d_pi.get(d, a, b);
        }
        for (size_t i = 0; i < Dim; ++i) {
          dt_phi.get(i, a, b) = logical_dt_phi.get(i, a, b);
          for (size_t d = 0; d < Dim; ++d) {
            dt_phi.get(i, a, b) -=
                face_mesh_velocity->get(d) * d_phi.get(d, i, a, b);
          }
        }
      }
    }
  } else {
    for (size_t storage_index = 0; storage_index < dt_pi.size();
         ++storage_index) {
      dt_spacetime_metric[storage_index].set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          make_not_null(&const_cast<DataVector&>(
              logical_dt_spacetime_metric[storage_index])));
      dt_pi[storage_index].set_data_ref(make_not_null(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          &const_cast<DataVector&>(logical_dt_pi[storage_index])));
    }
    for (size_t storage_index = 0; storage_index < dt_phi.size();
         ++storage_index) {
      dt_phi[storage_index].set_data_ref(make_not_null(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          &const_cast<DataVector&>(logical_dt_phi[storage_index])));
    }
  }

  auto& unit_interface_normal_vector =
      get<::Tags::TempI<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& four_index_constraint =
      get<::Tags::Tempiaa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& inverse_spatial_metric =
      get<::Tags::TempII<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& extrinsic_curvature =
      get<::Tags::Tempii<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& incoming_null_one_form =
      get<::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& outgoing_null_one_form =
      get<::Tags::Tempa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& incoming_null_vector =
      get<::Tags::TempA<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& outgoing_null_vector =
      get<::Tags::TempA<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& projection_ab =
      get<::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& projection_Ab =
      get<::Tags::TempAb<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& projection_AB =
      get<::Tags::TempAA<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_psi =
      get<::Tags::Tempaa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_zero =
      get<::Tags::Tempiaa<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_plus =
      get<::Tags::Tempaa<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_minus =
      get<::Tags::Tempaa<3, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& constraint_char_zero_plus =
      get<::Tags::Tempa<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& constraint_char_zero_minus =
      get<::Tags::Tempa<3, Dim, Frame::Inertial, DataVector>>(local_buffer);

  typename Tags::CharacteristicSpeeds<Dim, Frame::Inertial>::type char_speeds;

  auto& bc_dt_v_psi =
      get<::Tags::Tempaa<4, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_zero =
      get<::Tags::Tempiaa<3, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_plus =
      get<::Tags::Tempaa<5, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_minus =
      get<::Tags::Tempaa<6, Dim, Frame::Inertial, DataVector>>(local_buffer);

  compute_intermediate_vars(
      make_not_null(&unit_interface_normal_vector),
      make_not_null(&four_index_constraint),
      make_not_null(&inverse_spatial_metric),
      make_not_null(&extrinsic_curvature),
      make_not_null(&incoming_null_one_form),
      make_not_null(&outgoing_null_one_form),
      make_not_null(&incoming_null_vector),
      make_not_null(&outgoing_null_vector), make_not_null(&projection_ab),
      make_not_null(&projection_Ab), make_not_null(&projection_AB),
      make_not_null(&char_projected_rhs_dt_v_psi),
      make_not_null(&char_projected_rhs_dt_v_zero),
      make_not_null(&char_projected_rhs_dt_v_plus),
      make_not_null(&char_projected_rhs_dt_v_minus),
      make_not_null(&constraint_char_zero_plus),
      make_not_null(&constraint_char_zero_minus), make_not_null(&char_speeds),
      face_mesh_velocity, normal_covector, pi, phi, spacetime_metric, coords,
      gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
      spacetime_unit_normal_vector, spacetime_unit_normal_one_form,
      three_index_constraint, gauge_source, spacetime_deriv_gauge_source, dt_pi,
      dt_phi, dt_spacetime_metric, d_pi, d_phi, d_spacetime_metric);

  // Account for moving mesh: char speeds -> cher speeds - n_i v^i_g
  if (face_mesh_velocity.has_value()) {
    const auto radial_mesh_velocity =
        get(dot_product(normal_covector, *face_mesh_velocity));
    for (size_t a = 0; a < 4; ++a) {
      char_speeds.at(a) -= radial_mesh_velocity;
    }
  }

  // If no point on the boundary has any incoming characteristic, return here
  if (helpers::min_characteristic_speed(char_speeds) >= 0.) {
    std::fill(dt_spacetime_metric_correction->begin(),
              dt_spacetime_metric_correction->end(), 0.);
    std::fill(dt_pi_correction->begin(), dt_pi_correction->end(), 0.);
    std::fill(dt_phi_correction->begin(), dt_phi_correction->end(), 0.);
    return {};
  }

  Bjorhus::constraint_preserving_bjorhus_corrections_dt_v_psi(
      make_not_null(&bc_dt_v_psi), unit_interface_normal_vector,
      three_index_constraint, char_speeds);

  Bjorhus::constraint_preserving_bjorhus_corrections_dt_v_zero(
      make_not_null(&bc_dt_v_zero), unit_interface_normal_vector,
      four_index_constraint, char_speeds);

  // In order to set dt<V+> = 0, the correction term returned here must be
  // b_correction = -1*existing(dt<V+>), such that
  // final(dt<V+>) = existing(dt<V+>) + b_correction = 0
  for (size_t a = 0; a <= Dim; ++a) {
    for (size_t b = a; b <= Dim; ++b) {
      bc_dt_v_plus.get(a, b) = -char_projected_rhs_dt_v_plus.get(a, b);
    }
  }

  if (type_ == detail::ConstraintPreservingBjorhusType::ConstraintPreserving) {
    Bjorhus::constraint_preserving_bjorhus_corrections_dt_v_minus(
        make_not_null(&bc_dt_v_minus), gamma2, coords, incoming_null_one_form,
        outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
        projection_ab, projection_Ab, projection_AB,
        char_projected_rhs_dt_v_psi, char_projected_rhs_dt_v_minus,
        constraint_char_zero_plus, constraint_char_zero_minus, char_speeds);
  } else if (type_ == detail::ConstraintPreservingBjorhusType::
                          ConstraintPreservingPhysical) {
    Bjorhus::constraint_preserving_physical_bjorhus_corrections_dt_v_minus(
        make_not_null(&bc_dt_v_minus), gamma2, coords, normal_covector,
        unit_interface_normal_vector, spacetime_unit_normal_vector,
        incoming_null_one_form, outgoing_null_one_form, incoming_null_vector,
        outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
        inverse_spatial_metric, extrinsic_curvature, spacetime_metric,
        inverse_spacetime_metric, three_index_constraint,
        char_projected_rhs_dt_v_psi, char_projected_rhs_dt_v_minus,
        constraint_char_zero_plus, constraint_char_zero_minus, phi, d_phi, d_pi,
        char_speeds);
  } else {
    ERROR(
        "Failed to set dtVMinus. Input option must be one of "
        "ConstraintPreserving or ConstraintPreservingPhysical");
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

  // The boundary conditions here are imposed as corrections to the projections
  // of the right-hand-sides of the GH evolution equations (i.e. using Bjorhus'
  // method), and are written down in Eq. (63) - (65) of Lindblom et al (2005).
  // Now that we have calculated those corrections, we project them back as
  // corrections to dt<evolved variables>
  auto dt_evolved_vars = evolved_fields_from_characteristic_fields(
      gamma2, bc_dt_v_psi, bc_dt_v_zero, bc_dt_v_plus, bc_dt_v_minus,
      normal_covector);

  *dt_pi_correction = get<Tags::Pi<Dim, Frame::Inertial>>(dt_evolved_vars);
  *dt_phi_correction = get<Tags::Phi<Dim, Frame::Inertial>>(dt_evolved_vars);
  *dt_spacetime_metric_correction =
      get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
          dt_evolved_vars);

  if (face_mesh_velocity.has_value()) {
    const auto radial_mesh_velocity =
        get(dot_product(normal_covector, *face_mesh_velocity));
    // we use 1e-10 instead of 0 below to allow for purely tangentially
    // moving grids, eg a rotating sphere, with some leeway for
    // floating-point errors.
    if (max(radial_mesh_velocity) > 1.e-10) {
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
void ConstraintPreservingBjorhus<Dim>::compute_intermediate_vars(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        unit_interface_normal_vector,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        four_index_constraint,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        inverse_spatial_metric,
    const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
        extrinsic_curvature,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        incoming_null_one_form,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        outgoing_null_one_form,
    const gsl::not_null<tnsr::A<DataVector, Dim, Frame::Inertial>*>
        incoming_null_vector,
    const gsl::not_null<tnsr::A<DataVector, Dim, Frame::Inertial>*>
        outgoing_null_vector,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        projection_ab,
    const gsl::not_null<tnsr::Ab<DataVector, Dim, Frame::Inertial>*>
        projection_Ab,
    const gsl::not_null<tnsr::AA<DataVector, Dim, Frame::Inertial>*>
        projection_AB,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        char_projected_rhs_dt_v_psi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        char_projected_rhs_dt_v_zero,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        char_projected_rhs_dt_v_plus,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        char_projected_rhs_dt_v_minus,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        constraint_char_zero_plus,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        constraint_char_zero_minus,
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,

    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /* coords */,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::AA<DataVector, Dim, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::A<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::a<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_one_form,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
    const tnsr::a<DataVector, Dim, Frame::Inertial>& gauge_source,
    const tnsr::ab<DataVector, Dim, Frame::Inertial>&
        spacetime_deriv_gauge_source,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& dt_phi,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::ijaa<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& /* d_spacetime_metric */)
    const {
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataVector>,
                        ::Tags::Tempia<0, Dim, Frame::Inertial, DataVector>>>
      local_buffer(get_size(get<0>(normal_covector)), 0.);
  auto& one_over_lapse_sqrd =
      get(get<::Tags::TempScalar<0, DataVector>>(local_buffer));
  auto& two_index_constraint =
      get<::Tags::Tempia<0, Dim, Frame::Inertial, DataVector>>(local_buffer);

  one_over_lapse_sqrd = 1.0 / (get(lapse) * get(lapse));
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inverse_spatial_metric->get(i, j) =
          inverse_spacetime_metric.get(1 + i, 1 + j) +
          (shift.get(i) * shift.get(j) * one_over_lapse_sqrd);
    }
  }

  raise_or_lower_index(unit_interface_normal_vector, normal_covector,
                       *inverse_spatial_metric);

  GeneralizedHarmonic::extrinsic_curvature(
      extrinsic_curvature, spacetime_unit_normal_vector, pi, phi);

  if (LIKELY(Dim == 3)) {
    GeneralizedHarmonic::four_index_constraint(four_index_constraint, d_phi);
  } else if (UNLIKELY(Dim == 2)) {
    for (size_t a = 0; a <= Dim; ++a) {
      for (size_t b = 0; b <= Dim; ++b) {
        four_index_constraint->get(0, a, b) =
            d_phi.get(0, 1, a, b) - d_phi.get(1, 0, a, b);
        four_index_constraint->get(1, a, b) =
            -four_index_constraint->get(0, a, b);
      }
    }
  } else {
    std::fill(four_index_constraint->begin(), four_index_constraint->end(), 0.);
  }

  gr::interface_null_normal(incoming_null_one_form,
                            spacetime_unit_normal_one_form, normal_covector,
                            -1.);
  gr::interface_null_normal(outgoing_null_one_form,
                            spacetime_unit_normal_one_form, normal_covector,
                            1.);
  gr::interface_null_normal(incoming_null_vector, spacetime_unit_normal_vector,
                            *unit_interface_normal_vector, -1.);
  gr::interface_null_normal(outgoing_null_vector, spacetime_unit_normal_vector,
                            *unit_interface_normal_vector, 1.);

  gr::transverse_projection_operator(projection_ab, spacetime_metric,
                                     spacetime_unit_normal_one_form,
                                     normal_covector);
  gr::transverse_projection_operator(
      projection_Ab, spacetime_unit_normal_vector,
      spacetime_unit_normal_one_form, *unit_interface_normal_vector,
      normal_covector);
  gr::transverse_projection_operator(projection_AB, inverse_spacetime_metric,
                                     spacetime_unit_normal_vector,
                                     *unit_interface_normal_vector);

  const auto dt_char_fields = characteristic_fields(
      gamma2, *inverse_spatial_metric, dt_spacetime_metric, dt_pi, dt_phi,
      normal_covector);
  *char_projected_rhs_dt_v_psi =
      get<Tags::VSpacetimeMetric<Dim, Frame::Inertial>>(dt_char_fields);
  *char_projected_rhs_dt_v_zero =
      get<Tags::VZero<Dim, Frame::Inertial>>(dt_char_fields);
  *char_projected_rhs_dt_v_plus =
      get<Tags::VPlus<Dim, Frame::Inertial>>(dt_char_fields);
  *char_projected_rhs_dt_v_minus =
      get<Tags::VMinus<Dim, Frame::Inertial>>(dt_char_fields);

  // c^{\hat{0}-}_a = F_a + n^k C_{ka}
  GeneralizedHarmonic::two_index_constraint(
      make_not_null(&two_index_constraint), spacetime_deriv_gauge_source,
      spacetime_unit_normal_one_form, spacetime_unit_normal_vector,
      *inverse_spatial_metric, inverse_spacetime_metric, pi, phi, d_pi, d_phi,
      gamma2, three_index_constraint);
  f_constraint(constraint_char_zero_plus, gauge_source,
               spacetime_deriv_gauge_source, spacetime_unit_normal_one_form,
               spacetime_unit_normal_vector, *inverse_spatial_metric,
               inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
               three_index_constraint);
  for (size_t a = 0; a < Dim + 1; ++a) {
    constraint_char_zero_minus->get(a) = constraint_char_zero_plus->get(a);
    for (size_t i = 0; i < Dim; ++i) {
      constraint_char_zero_plus->get(a) -=
          unit_interface_normal_vector->get(i) * two_index_constraint.get(i, a);
      constraint_char_zero_minus->get(a) +=
          unit_interface_normal_vector->get(i) * two_index_constraint.get(i, a);
    }
  }

  characteristic_speeds(char_speeds, gamma1, lapse, shift, normal_covector,
                        face_mesh_velocity);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID ConstraintPreservingBjorhus<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class ConstraintPreservingBjorhus<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic::BoundaryConditions
