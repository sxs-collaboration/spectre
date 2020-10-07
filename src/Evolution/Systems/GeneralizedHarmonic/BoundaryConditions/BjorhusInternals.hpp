// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace GeneralizedHarmonic {
namespace Actions {
namespace BoundaryConditions_detail {
class BjorhusIntermediatesComputer {
 public:
  struct internal_tags {
    // null vectors and forms
    template <size_t VolumeDim, typename DataType = DataVector>
    using outgoing_null_one_form =
        ::Tags::Tempa<0, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using incoming_null_one_form =
        ::Tags::Tempa<1, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using outgoing_null_vector =
        ::Tags::TempA<2, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using incoming_null_vector =
        ::Tags::TempA<3, VolumeDim, Frame::Inertial, DataType>;
    // boundary normals
    template <size_t VolumeDim, typename DataType = DataVector>
    using interface_normal_one_form =
        ::Tags::Tempa<4, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using interface_normal_vector =
        ::Tags::TempA<5, VolumeDim, Frame::Inertial, DataType>;
    template <typename DataType = DataVector>
    using interface_normal_dot_shift = ::Tags::TempScalar<8, DataType>;
    // projection tensors
    template <size_t VolumeDim, typename DataType = DataVector>
    using projection_AB =
        ::Tags::TempAA<9, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using projection_ab =
        ::Tags::Tempaa<10, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using projection_Ab =
        ::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataType>;
    // Characteristics
    template <typename DataType = DataVector>
    using char_speed_vpsi = ::Tags::TempScalar<12, DataType>;
    template <typename DataType = DataVector>
    using char_speed_vzero = ::Tags::TempScalar<13, DataType>;
    template <typename DataType = DataVector>
    using char_speed_vplus = ::Tags::TempScalar<14, DataType>;
    template <typename DataType = DataVector>
    using char_speed_vminus = ::Tags::TempScalar<15, DataType>;
    // Constraints
    template <size_t VolumeDim, typename DataType = DataVector>
    using constraint_char_zero_minus =
        ::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using constraint_char_zero_plus =
        ::Tags::Tempa<34, VolumeDim, Frame::Inertial, DataType>;
    // Memory for BCs
    template <size_t VolumeDim, typename DataType = DataVector>
    using bc_dt_u_psi =
        ::Tags::Tempaa<27, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using bc_dt_u_zero =
        ::Tags::Tempiaa<28, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using bc_dt_u_plus =
        ::Tags::Tempaa<29, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using bc_dt_u_minus =
        ::Tags::Tempaa<30, VolumeDim, Frame::Inertial, DataType>;
    // Derivs of evolved vars
    template <size_t VolumeDim, typename DataType = DataVector>
    using deriv_spacetime_metric =
        ::Tags::Tempiaa<31, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using deriv_pi = ::Tags::Tempiaa<32, VolumeDim, Frame::Inertial, DataType>;
    template <size_t VolumeDim, typename DataType = DataVector>
    using deriv_phi =
        ::Tags::Tempijaa<33, VolumeDim, Frame::Inertial, DataType>;
  };

  template <size_t VolumeDim>
  using intermediate_vars = tmpl::list<
      internal_tags::outgoing_null_one_form<VolumeDim, DataVector>,
      internal_tags::incoming_null_one_form<VolumeDim, DataVector>,
      internal_tags::outgoing_null_vector<VolumeDim, DataVector>,
      internal_tags::incoming_null_vector<VolumeDim, DataVector>,
      internal_tags::interface_normal_one_form<VolumeDim, DataVector>,
      internal_tags::interface_normal_vector<VolumeDim, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial, DataVector>,
      internal_tags::interface_normal_dot_shift<DataVector>,
      internal_tags::projection_AB<VolumeDim, DataVector>,
      internal_tags::projection_ab<VolumeDim, DataVector>,
      internal_tags::projection_Ab<VolumeDim, DataVector>,
      internal_tags::char_speed_vpsi<DataVector>,
      internal_tags::char_speed_vzero<DataVector>,
      internal_tags::char_speed_vplus<DataVector>,
      internal_tags::char_speed_vminus<DataVector>,
      internal_tags::constraint_char_zero_minus<VolumeDim, DataVector>,
      internal_tags::constraint_char_zero_plus<VolumeDim, DataVector>,
      Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>,
      Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::ExtrinsicCurvature<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
      Tags::VSpacetimeMetric<VolumeDim, Frame::Inertial>,
      Tags::VZero<VolumeDim, Frame::Inertial>,
      Tags::VPlus<VolumeDim, Frame::Inertial>,
      Tags::VMinus<VolumeDim, Frame::Inertial>, Tags::ConstraintGamma2,
      internal_tags::bc_dt_u_psi<VolumeDim, DataVector>,
      internal_tags::bc_dt_u_zero<VolumeDim, DataVector>,
      internal_tags::bc_dt_u_plus<VolumeDim, DataVector>,
      internal_tags::bc_dt_u_minus<VolumeDim, DataVector>,
      internal_tags::deriv_spacetime_metric<VolumeDim, DataVector>,
      internal_tags::deriv_pi<VolumeDim, DataVector>,
      internal_tags::deriv_phi<VolumeDim, DataVector>,
      domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>;

 private:
  TempBuffer<intermediate_vars<3>> buffer_;

 public:
  explicit BjorhusIntermediatesComputer(size_t used_for_size) noexcept
      : buffer_(used_for_size) {}

  template <typename Tag>
  const typename Tag::type& get_var(const Tag& /* tag */) const noexcept {
    return get<Tag>(buffer_);
  }

  template <typename Tag>
  typename Tag::type& get_var(const Tag& /* tag */) noexcept {
    return get<Tag>(buffer_);
  }

  template <typename Tag>
  void set_var(const typename Tag::type& var_data,
               const Tag& /* tag */) noexcept {
    get<Tag>(buffer_) = var_data;
  }

  SPECTRE_ALWAYS_INLINE decltype(buffer_)& get_buffer() noexcept {
    return buffer_;
  }

  // \brief This function computes intermediate variables needed for
  // Bjorhus-type constraint preserving boundary conditions for the
  // GeneralizedHarmonic system
  template <size_t VolumeDim, typename DbTags, typename VarsTagsList,
            typename DtVarsTagsList>
  void compute_vars(
      const db::DataBox<DbTags>& box, const Direction<VolumeDim>& direction,
      const size_t& dimension,
      const typename domain::Tags::Mesh<VolumeDim>::type& mesh,
      const Variables<VarsTagsList>& /* vars */,
      const Variables<DtVarsTagsList>& dt_vars,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_one_form,
      const typename Tags::CharacteristicSpeeds<
          VolumeDim, Frame::Inertial>::type& char_speeds) noexcept {
    //
    // NOTE: variable names below closely follow \cite Lindblom2005qh
    //
    // 1) Extract quantities from databox that are needed to compute
    // intermediate variables
    using tags_needed_on_slice = tmpl::list<
        gr::Tags::Lapse<DataVector>,
        gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
        gr::Tags::SpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
        gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
        gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial,
                                         DataVector>,
        gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial, DataVector>,
        gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
        gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial,
                                         DataVector>,
        gr::Tags::ExtrinsicCurvature<VolumeDim, Frame::Inertial, DataVector>,
        ::Tags::deriv<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial>,
                      tmpl::size_t<VolumeDim>, Frame::Inertial>,
        ::Tags::deriv<Tags::Pi<VolumeDim, Frame::Inertial>,
                      tmpl::size_t<VolumeDim>, Frame::Inertial>,
        ::Tags::deriv<Tags::Phi<VolumeDim, Frame::Inertial>,
                      tmpl::size_t<VolumeDim>, Frame::Inertial>,
        Tags::ConstraintGamma2,
        Tags::TwoIndexConstraint<VolumeDim, Frame::Inertial>,
        Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>,
        Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>,
        Tags::FConstraint<VolumeDim, Frame::Inertial>,
        domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>;
    const auto vars_on_this_slice = db::data_on_slice(
        box, mesh.extents(), dimension,
        index_to_slice_at(mesh.extents(), direction), tags_needed_on_slice{});

    // 2) name quantities as its just easier
    const auto& shift =
        get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(
            vars_on_this_slice);
    const auto& inverse_spatial_metric = get<
        gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>>(
        vars_on_this_slice);
    const auto& spacetime_normal_one_form =
        get<gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial,
                                             DataVector>>(vars_on_this_slice);
    const auto& spacetime_normal_vector =
        get<gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial,
                                            DataVector>>(vars_on_this_slice);
    const auto& spacetime_metric =
        get<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>(
            vars_on_this_slice);
    const auto& inverse_spacetime_metric =
        get<gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial,
                                             DataVector>>(vars_on_this_slice);

    const auto& gamma2 = get<Tags::ConstraintGamma2>(vars_on_this_slice);
    const auto& two_index_constraint =
        get<Tags::TwoIndexConstraint<VolumeDim, Frame::Inertial>>(
            vars_on_this_slice);
    const auto& f_constraint =
        get<Tags::FConstraint<VolumeDim, Frame::Inertial>>(vars_on_this_slice);
    // storage for DT<UChar> = CharProjection(dt<U>)
    const auto& rhs_dt_psi = get<::Tags::dt<
        gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>>(
        dt_vars);
    const auto& rhs_dt_pi =
        get<::Tags::dt<Tags::Pi<VolumeDim, Frame::Inertial>>>(dt_vars);
    const auto& rhs_dt_phi =
        get<::Tags::dt<Tags::Phi<VolumeDim, Frame::Inertial>>>(dt_vars);
    const auto char_projected_dt_u = characteristic_fields(
        gamma2, inverse_spatial_metric, rhs_dt_psi, rhs_dt_pi, rhs_dt_phi,
        unit_interface_normal_one_form);

    // 3) Extract variable storage out of the buffer now
    // timelike and spacelike SPACETIME vectors, l^a and k^a
    auto& local_outgoing_null_one_form =
        get<internal_tags::outgoing_null_one_form<VolumeDim, DataVector>>(
            buffer_);
    auto& local_incoming_null_one_form =
        get<internal_tags::incoming_null_one_form<VolumeDim, DataVector>>(
            buffer_);
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    auto& local_outgoing_null_vector =
        get<internal_tags::outgoing_null_vector<VolumeDim, DataVector>>(
            buffer_);
    auto& local_incoming_null_vector =
        get<internal_tags::incoming_null_vector<VolumeDim, DataVector>>(
            buffer_);
    // SPACETIME form of interface normal (vector and oneform)
    auto& local_interface_normal_one_form =
        get<internal_tags::interface_normal_one_form<VolumeDim, DataVector>>(
            buffer_);
    auto& local_interface_normal_vector =
        get<internal_tags::interface_normal_vector<VolumeDim, DataVector>>(
            buffer_);
    // spacetime null form t_a and vector t^a
    get<gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial,
                                         DataVector>>(buffer_) =
        get<gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial,
                                             DataVector>>(vars_on_this_slice);
    get<gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial,
                                        DataVector>>(buffer_) =
        get<gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial,
                                            DataVector>>(vars_on_this_slice);
    // interface normal dot shift: n_k N^k
    auto& interface_normal_dot_shift =
        get<internal_tags::interface_normal_dot_shift<DataVector>>(buffer_);
    // spacetime projection operator P_ab, P^ab, and P^a_b
    auto& local_projection_AB =
        get<internal_tags::projection_AB<VolumeDim, DataVector>>(buffer_);
    auto& local_projection_ab =
        get<internal_tags::projection_ab<VolumeDim, DataVector>>(buffer_);
    auto& local_projection_Ab =
        get<internal_tags::projection_Ab<VolumeDim, DataVector>>(buffer_);
    // 4.4) Characteristic speeds
    get(get<internal_tags::char_speed_vpsi<DataVector>>(buffer_)) =
        char_speeds.at(0);
    get(get<internal_tags::char_speed_vzero<DataVector>>(buffer_)) =
        char_speeds.at(1);
    get(get<internal_tags::char_speed_vplus<DataVector>>(buffer_)) =
        char_speeds.at(2);
    get(get<internal_tags::char_speed_vminus<DataVector>>(buffer_)) =
        char_speeds.at(3);
    // constraint characteristics
    auto& local_constraint_char_zero_minus =
        get<internal_tags::constraint_char_zero_minus<VolumeDim, DataVector>>(
            buffer_);
    auto& local_constraint_char_zero_plus =
        get<internal_tags::constraint_char_zero_plus<VolumeDim, DataVector>>(
            buffer_);
    // 4.6) c^\hat{3}_{jab} = C_{jab} = \partial_j\psi_{ab} - \Phi_{jab}
    get<Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>>(buffer_) =
        get<Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>>(
            vars_on_this_slice);
    // 4.7) c^\hat{4}_{ijab} = C_{ijab}
    get<Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>>(buffer_) =
        get<Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>>(
            vars_on_this_slice);
    // lapse, shift and inverse spatial_metric
    get<gr::Tags::Lapse<DataVector>>(buffer_) =
        get<gr::Tags::Lapse<DataVector>>(vars_on_this_slice);
    get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(buffer_) =
        get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(
            vars_on_this_slice);
    get<gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>>(
        buffer_) =
        get<gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial,
                                           DataVector>>(vars_on_this_slice);
    get<gr::Tags::ExtrinsicCurvature<VolumeDim, Frame::Inertial, DataVector>>(
        buffer_) =
        get<gr::Tags::ExtrinsicCurvature<VolumeDim, Frame::Inertial,
                                         DataVector>>(vars_on_this_slice);
    get<gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial,
                                         DataVector>>(buffer_) =
        get<gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial,
                                             DataVector>>(vars_on_this_slice);
    // Characteristic projected time derivatives of evolved fields
    get<Tags::VSpacetimeMetric<VolumeDim, Frame::Inertial>>(buffer_) =
        get<Tags::VSpacetimeMetric<VolumeDim, Frame::Inertial>>(
            char_projected_dt_u);
    get<Tags::VZero<VolumeDim, Frame::Inertial>>(buffer_) =
        get<Tags::VZero<VolumeDim, Frame::Inertial>>(char_projected_dt_u);
    get<Tags::VPlus<VolumeDim, Frame::Inertial>>(buffer_) =
        get<Tags::VPlus<VolumeDim, Frame::Inertial>>(char_projected_dt_u);
    get<Tags::VMinus<VolumeDim, Frame::Inertial>>(buffer_) =
        get<Tags::VMinus<VolumeDim, Frame::Inertial>>(char_projected_dt_u);
    // Spatial derivatives of evolved variables: Psi, Pi and Phi
    get<internal_tags::deriv_spacetime_metric<VolumeDim, DataVector>>(buffer_) =
        get<::Tags::deriv<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial>,
                          tmpl::size_t<VolumeDim>, Frame::Inertial>>(
            vars_on_this_slice);
    get<internal_tags::deriv_pi<VolumeDim, DataVector>>(buffer_) =
        get<::Tags::deriv<Tags::Pi<VolumeDim, Frame::Inertial>,
                          tmpl::size_t<VolumeDim>, Frame::Inertial>>(
            vars_on_this_slice);
    get<internal_tags::deriv_phi<VolumeDim, DataVector>>(buffer_) =
        get<::Tags::deriv<Tags::Phi<VolumeDim, Frame::Inertial>,
                          tmpl::size_t<VolumeDim>, Frame::Inertial>>(
            vars_on_this_slice);
    // Constraint damping parameters
    get<Tags::ConstraintGamma2>(buffer_) =
        get<Tags::ConstraintGamma2>(vars_on_this_slice);

    // Coordinates
    get<domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>(buffer_) =
        get<domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>(
            vars_on_this_slice);

    // 4) Compute intermediate variables now
    // 4.1) Spacetime form of interface normal (vector and oneform)
    const auto unit_interface_normal_vector = raise_or_lower_index(
        unit_interface_normal_one_form, inverse_spatial_metric);
    get<0>(local_interface_normal_one_form) = 0.;
    get<0>(local_interface_normal_vector) = 0.;
    for (size_t i = 0; i < VolumeDim; ++i) {
      local_interface_normal_one_form.get(1 + i) =
          unit_interface_normal_one_form.get(i);
      local_interface_normal_vector.get(1 + i) =
          unit_interface_normal_vector.get(i);
    }
    // 4.2) timelike and spacelike SPACETIME vectors, l^a and k^a,
    //      without (1/sqrt(2))
    for (size_t a = 0; a < VolumeDim + 1; ++a) {
      local_outgoing_null_one_form.get(a) =
          spacetime_normal_one_form.get(a) +
          local_interface_normal_one_form.get(a);
      local_incoming_null_one_form.get(a) =
          spacetime_normal_one_form.get(a) -
          local_interface_normal_one_form.get(a);
      local_outgoing_null_vector.get(a) =
          spacetime_normal_vector.get(a) + local_interface_normal_vector.get(a);
      local_incoming_null_vector.get(a) =
          spacetime_normal_vector.get(a) - local_interface_normal_vector.get(a);
    }
    //       interface_normal_dot_shift = n_i N^i
    for (size_t i = 0; i < VolumeDim; ++i) {
      get(interface_normal_dot_shift) =
          shift.get(i) * local_interface_normal_one_form.get(i + 1);
    }
    // 4.3) Spacetime projection operators P_ab, P^ab and P^a_b
    for (size_t a = 0; a < VolumeDim + 1; ++a) {
      for (size_t b = 0; b < VolumeDim + 1; ++b) {
        local_projection_ab.get(a, b) =
            spacetime_metric.get(a, b) +
            spacetime_normal_one_form.get(a) *
                spacetime_normal_one_form.get(b) -
            local_interface_normal_one_form.get(a) *
                local_interface_normal_one_form.get(b);
        local_projection_Ab.get(a, b) =
            spacetime_normal_one_form.get(a) * spacetime_normal_vector.get(b) -
            local_interface_normal_one_form.get(a) *
                local_interface_normal_vector.get(b);
        if (UNLIKELY(a == b)) {
          local_projection_Ab.get(a, b) += 1.;
        }
        local_projection_AB.get(a, b) =
            inverse_spacetime_metric.get(a, b) +
            spacetime_normal_vector.get(a) * spacetime_normal_vector.get(b) -
            local_interface_normal_vector.get(a) *
                local_interface_normal_vector.get(b);
      }
    }
    // 4.5) c^{\hat{0}-}_a = F_a + n^k C_{ka}
    for (size_t a = 0; a < VolumeDim + 1; ++a) {
      local_constraint_char_zero_minus.get(a) = f_constraint.get(a);
      local_constraint_char_zero_plus.get(a) = f_constraint.get(a);
      for (size_t i = 0; i < VolumeDim; ++i) {
        local_constraint_char_zero_minus.get(a) +=
            unit_interface_normal_vector.get(i) *
            two_index_constraint.get(i, a);
        local_constraint_char_zero_plus.get(a) -=
            unit_interface_normal_vector.get(i) *
            two_index_constraint.get(i, a);
      }
    }
  }
};
}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace GeneralizedHarmonic
