// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
namespace detail {
template <evolution::BoundaryConditions::Type GhBcType,
          evolution::BoundaryConditions::Type ValenciaBcType>
struct UnionOfBcTypes {
  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::GhostAndTimeDerivative;
};

template <>
struct UnionOfBcTypes<evolution::BoundaryConditions::Type::Ghost,
                      evolution::BoundaryConditions::Type::Ghost> {
  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;
};

template <>
struct UnionOfBcTypes<evolution::BoundaryConditions::Type::TimeDerivative,
                      evolution::BoundaryConditions::Type::TimeDerivative> {
  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;
};

template <evolution::BoundaryConditions::Type GhBcType>
struct UnionOfBcTypes<GhBcType, evolution::BoundaryConditions::Type::Outflow> {
  static_assert(GhBcType == evolution::BoundaryConditions::Type::Outflow,
                "If either boundary condition in `ProductOfConditions` has "
                "`Type::Outflow`, both must have `Type::Outflow`");
};

template <evolution::BoundaryConditions::Type ValenciaBcType>
struct UnionOfBcTypes<evolution::BoundaryConditions::Type::Outflow,
                      ValenciaBcType> {
  static_assert(ValenciaBcType == evolution::BoundaryConditions::Type::Outflow,
                "If either boundary condition in `ProductOfConditions` has "
                "`Type::Outflow`, both must have `Type::Outflow`");
};

template <>
struct UnionOfBcTypes<evolution::BoundaryConditions::Type::Outflow,
                      evolution::BoundaryConditions::Type::Outflow> {
  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Outflow;
};

struct ValenciaDoNothingGhostCondition {
  using dg_interior_evolved_variables_tags =
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list;

  using dg_interior_temporary_tags = tmpl::push_back<
      db::wrap_tags_in<::Tags::Flux,
                       typename grmhd::ValenciaDivClean::System::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>;

  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> tilde_d,
      const gsl::not_null<Scalar<DataVector>*> tilde_tau,
      const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      const gsl::not_null<Scalar<DataVector>*> tilde_phi,

      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_d_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_tau_flux,
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          tilde_s_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          tilde_b_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_phi_flux,

      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<
          tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,

      const Scalar<DataVector>& interior_tilde_d,
      const Scalar<DataVector>& interior_tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_b,
      const Scalar<DataVector>& interior_tilde_phi,

      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_d_flux,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_tau_flux,
      const tnsr::Ij<DataVector, 3, Frame::Inertial>& interior_tilde_s_flux,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& interior_tilde_b_flux,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_tilde_phi_flux,

      const tnsr::aa<DataVector, 3, Frame::Inertial>&
          interior_spacetime_metric) noexcept {
    *tilde_d = interior_tilde_d;
    *tilde_tau = interior_tilde_tau;
    *tilde_s = interior_tilde_s;
    *tilde_b = interior_tilde_b;
    *tilde_phi = interior_tilde_phi;

    *tilde_d_flux = interior_tilde_d_flux;
    *tilde_tau_flux = interior_tilde_tau_flux;
    *tilde_s_flux = interior_tilde_s_flux;
    *tilde_b_flux = interior_tilde_b_flux;
    *tilde_phi_flux = interior_tilde_phi_flux;

    const auto spatial_metric = gr::spatial_metric(interior_spacetime_metric);
    *inv_spatial_metric = determinant_and_inverse(spatial_metric).second;
    gr::shift(shift, interior_spacetime_metric, *inv_spatial_metric);
    gr::lapse(lapse, *shift, interior_spacetime_metric);
    return {};
  }
};

struct GhDoNothingGhostCondition {
  using dg_interior_evolved_variables_tags =
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list;

  using dg_interior_temporary_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>;

  std::optional<std::string> dg_ghost(
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
          spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<
          tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,
      const Scalar<DataVector>& interior_gamma1,
      const Scalar<DataVector>& interior_gamma2) noexcept {
    *gamma1 = interior_gamma1;
    *gamma2 = interior_gamma2;
    *spacetime_metric = interior_spacetime_metric;
    *pi = interior_pi;
    *phi = interior_phi;

    const auto spatial_metric = gr::spatial_metric(interior_spacetime_metric);
    *inv_spatial_metric = determinant_and_inverse(spatial_metric).second;
    gr::shift(shift, interior_spacetime_metric, *inv_spatial_metric);
    gr::lapse(lapse, *shift, interior_spacetime_metric);
    return {};
  }
};

// Implementation for expanding the combination of packs for full compatibility
// with all possible combination of tag lists that can be used for constructing
// `BoundaryCondition` structs.
template <
    typename DerivedGhCondition, typename DerivedValenciaCondition,
    typename GhEvolvedTagList, typename ValenciaEvolvedTagList,
    typename GhFluxTagList, typename ValenicaFluxTagList,
    typename GhInteriorEvolvedTagList, typename ValenciaInteriorEvolvedTagList,
    typename DeduplicatedInteriorEvolvedTagList,
    typename GhInteriorPrimitiveTagList, typename ValenciaInteriorPrimitiveTgs,
    typename GhInteriorTempTagList, typename ValenciaInteriorTempTagList,
    typename DeduplicatedTempTagList, typename GhInteriorDtTagList,
    typename ValenciaInteriorDtTagList, typename GhInteriorDerivTagList,
    typename ValenciaInteriorDerivTagList, typename GhGridlessTagList,
    typename ValenciaGridlessTagList, typename DeduplicatedGridlessTagList>
struct ProductOfConditionsImpl;

template <
    typename DerivedGhCondition, typename DerivedValenciaCondition,
    typename... GhEvolvedTags, typename... ValenciaEvolvedTags,
    typename... GhFluxTags, typename... ValenciaFluxTags,
    typename... GhInteriorEvolvedTags, typename... ValenciaInteriorEvolvedTags,
    typename... DeduplicatedInteriorEvolvedTags,
    typename... GhInteriorPrimitiveTags,
    typename... ValenciaInteriorPrimitiveTags, typename... GhInteriorTempTags,
    typename... ValenciaInteriorTempTags, typename... DeduplicatedTempTags,
    typename... GhInteriorDtTags, typename... ValenciaInteriorDtTags,
    typename... GhInteriorDerivTags, typename... ValenciaInteriorDerivTags,
    typename... GhGridlessTags, typename... ValenciaGridlessTags,
    typename... DeduplicatedGridlessTags>
struct ProductOfConditionsImpl<
    DerivedGhCondition, DerivedValenciaCondition, tmpl::list<GhEvolvedTags...>,
    tmpl::list<ValenciaEvolvedTags...>, tmpl::list<GhFluxTags...>,
    tmpl::list<ValenciaFluxTags...>, tmpl::list<GhInteriorEvolvedTags...>,
    tmpl::list<ValenciaInteriorEvolvedTags...>,
    tmpl::list<DeduplicatedInteriorEvolvedTags...>,
    tmpl::list<GhInteriorPrimitiveTags...>,
    tmpl::list<ValenciaInteriorPrimitiveTags...>,
    tmpl::list<GhInteriorTempTags...>, tmpl::list<ValenciaInteriorTempTags...>,
    tmpl::list<DeduplicatedTempTags...>, tmpl::list<GhInteriorDtTags...>,
    tmpl::list<ValenciaInteriorDtTags...>, tmpl::list<GhInteriorDerivTags...>,
    tmpl::list<ValenciaInteriorDerivTags...>, tmpl::list<GhGridlessTags...>,
    tmpl::list<ValenciaGridlessTags...>,
    tmpl::list<DeduplicatedGridlessTags...>> {
  // In the current setup, we aren't given type information about the possible
  // arguments to `BoundaryCorrection`s directly, so we just need to support all
  // possibilities of (GH BoundaryCorrection)x(Valencia BoundaryCorrection) with
  // explicit overloads of `dg_ghost`.
  // This can be solved with (more) template logic instead if in the future
  // `BoundaryCondition`s can supply stronger type constraints

  template <typename... GridlessVariables>
  static std::optional<std::string> dg_ghost(
      const DerivedGhCondition& gh_condition,
      const DerivedValenciaCondition& valencia_condition,

      const gsl::not_null<typename GhEvolvedTags::type*>... gh_variables,
      const gsl::not_null<
          typename ValenciaEvolvedTags::type*>... valencia_variables,

      const gsl::not_null<typename GhFluxTags::type*>... gh_fluxes,
      const gsl::not_null<typename ValenciaFluxTags::type*>... valencia_fluxes,

      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, 3_st, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<tnsr::I<DataVector, 3_st, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3_st, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3_st, Frame::Inertial>& normal_vector,

      const typename DeduplicatedInteriorEvolvedTags::
          type&... int_evolved_variables,

      const typename GhInteriorPrimitiveTags::type&... gh_int_prim_variables,
      const typename ValenciaInteriorPrimitiveTags::
          type&... valencia_int_prim_variables,

      const typename DeduplicatedTempTags::type&... temp_variables,

      const typename GhInteriorDtTags::type&... gh_int_dt_variables,
      const typename ValenciaInteriorDtTags::type&... valencia_int_dt_variables,

      const typename GhInteriorDerivTags::type&... gh_int_deriv_variables,
      const typename ValenciaInteriorDerivTags::
          type&... valencia_int_deriv_variables,

      const GridlessVariables&... gridless_variables) noexcept {
    using gridless_tags_and_types =
        tmpl::map<tmpl::pair<DeduplicatedGridlessTags, GridlessVariables>...>;

    tuples::TaggedTuple<
        Tags::detail::TemporaryReference<
            DeduplicatedGridlessTags,
            tmpl::at<gridless_tags_and_types, DeduplicatedGridlessTags>>...,
        Tags::detail::TemporaryReference<DeduplicatedTempTags>...,
        Tags::detail::TemporaryReference<DeduplicatedInteriorEvolvedTags>...>
        shuffle_refs{gridless_variables..., temp_variables...,
                     int_evolved_variables...};

    std::optional<std::string> gh_string{};
    if constexpr (DerivedGhCondition::bc_type ==
                      evolution::BoundaryConditions::Type::Ghost or
                  DerivedGhCondition::bc_type ==
                      evolution::BoundaryConditions::Type::
                          GhostAndTimeDerivative) {
      gh_string = gh_condition.dg_ghost(
          gh_variables..., gh_fluxes..., gamma1, gamma2, lapse, shift,
          inv_spatial_metric, face_mesh_velocity, normal_covector,
          normal_vector,
          tuples::get<Tags::detail::TemporaryReference<GhInteriorEvolvedTags>>(
              shuffle_refs)...,
          gh_int_prim_variables...,
          tuples::get<Tags::detail::TemporaryReference<GhInteriorTempTags>>(
              shuffle_refs)...,
          gh_int_dt_variables..., gh_int_deriv_variables...,
          tuples::get<Tags::detail::TemporaryReference<
              GhGridlessTags,
              tmpl::at<gridless_tags_and_types, GhGridlessTags>>>(
              shuffle_refs)...);
    } else {
      GhDoNothingGhostCondition{}.dg_ghost(
          gh_variables..., gh_fluxes..., gamma1, gamma2, lapse, shift,
          inv_spatial_metric, face_mesh_velocity, normal_covector,
          normal_vector,
          tuples::get<Tags::detail::TemporaryReference<GhEvolvedTags>>(
              shuffle_refs)...,
          tuples::get<Tags::detail::TemporaryReference<
              ::GeneralizedHarmonic::ConstraintDamping::Tags::
                  ConstraintGamma1>>(shuffle_refs),
          tuples::get<Tags::detail::TemporaryReference<
              ::GeneralizedHarmonic::ConstraintDamping::Tags::
                  ConstraintGamma2>>(shuffle_refs));
    }
    std::optional<std::string> valencia_string{};
    if constexpr (DerivedValenciaCondition::bc_type ==
                      evolution::BoundaryConditions::Type::Ghost or
                  DerivedValenciaCondition::bc_type ==
                      evolution::BoundaryConditions::Type::
                          GhostAndTimeDerivative) {
      valencia_string = valencia_condition.dg_ghost(
          valencia_variables..., valencia_fluxes..., lapse, shift,
          inv_spatial_metric, face_mesh_velocity, normal_covector,
          normal_vector,
          tuples::get<
              Tags::detail::TemporaryReference<ValenciaInteriorEvolvedTags>>(
              shuffle_refs)...,
          valencia_int_prim_variables...,
          tuples::get<
              Tags::detail::TemporaryReference<ValenciaInteriorTempTags>>(
              shuffle_refs)...,
          valencia_int_dt_variables..., valencia_int_deriv_variables...,
          tuples::get<Tags::detail::TemporaryReference<
              ValenciaGridlessTags,
              tmpl::at<gridless_tags_and_types, ValenciaGridlessTags>>>(
              shuffle_refs)...);
    } else {
      ValenciaDoNothingGhostCondition{}.dg_ghost(
          valencia_variables..., valencia_fluxes..., gamma1, gamma2, lapse,
          shift, inv_spatial_metric, face_mesh_velocity, normal_covector,
          normal_vector,
          tuples::get<Tags::detail::TemporaryReference<ValenciaEvolvedTags>>(
              shuffle_refs)...,
          tuples::get<Tags::detail::TemporaryReference<ValenciaFluxTags>>(
              shuffle_refs)...,
          tuples::get<Tags::detail::TemporaryReference<
              gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>>(
              shuffle_refs));
    }
    if (not gh_string.has_value()) {
      return valencia_string;
    }
    if (not valencia_string.has_value()) {
      return gh_string;
    }
    return gh_string.value() + ";" + valencia_string.value();
  }

  template <typename... GridlessVariables>
  static std::optional<std::string> dg_outflow(
      const DerivedGhCondition& gh_condition,
      const DerivedValenciaCondition& valencia_condition,

      const std::optional<tnsr::I<DataVector, 3_st, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3_st, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3_st, Frame::Inertial>& normal_vector,

      const typename DeduplicatedInteriorEvolvedTags::
          type&... int_evolved_variables,

      const typename GhInteriorPrimitiveTags::type&... gh_int_prim_variables,
      const typename ValenciaInteriorPrimitiveTags::
          type&... valencia_int_prim_variables,

      const typename DeduplicatedTempTags::type&... temp_variables,

      const typename GhInteriorDtTags::type&... gh_int_dt_variables,
      const typename ValenciaInteriorDtTags::type&... valencia_int_dt_variables,

      const typename GhInteriorDerivTags::type&... gh_int_deriv_variables,
      const typename ValenciaInteriorDerivTags::
          type&... valencia_int_deriv_variables,

      const GridlessVariables&... gridless_variables) noexcept {
    using gridless_tags_and_types =
        tmpl::map<tmpl::pair<DeduplicatedGridlessTags, GridlessVariables>...>;

    tuples::TaggedTuple<
        Tags::detail::TemporaryReference<
            DeduplicatedGridlessTags,
            tmpl::at<gridless_tags_and_types, DeduplicatedGridlessTags>>...,
        Tags::detail::TemporaryReference<DeduplicatedTempTags>...,
        Tags::detail::TemporaryReference<DeduplicatedInteriorEvolvedTags>...>
        shuffle_refs{gridless_variables..., temp_variables...,
                     int_evolved_variables...};
    // outflow condition is only valid if both boundary conditions are outflow,
    // so we directly apply both. A static_assert elsewhere is triggered if only
    // one boundary condition is outflow.
    auto gh_string = gh_condition.dg_outflow(
        face_mesh_velocity, normal_covector, normal_vector,
        tuples::get<Tags::detail::TemporaryReference<GhInteriorEvolvedTags>>(
            shuffle_refs)...,
        gh_int_prim_variables...,
        tuples::get<Tags::detail::TemporaryReference<GhInteriorTempTags>>(
            shuffle_refs)...,
        gh_int_dt_variables..., gh_int_deriv_variables...,
        tuples::get<Tags::detail::TemporaryReference<
            GhGridlessTags, tmpl::at<gridless_tags_and_types, GhGridlessTags>>>(
            shuffle_refs)...);
    auto valencia_string = valencia_condition.dg_outflow(
        face_mesh_velocity, normal_covector, normal_vector,
        tuples::get<
            Tags::detail::TemporaryReference<ValenciaInteriorEvolvedTags>>(
            shuffle_refs)...,
        valencia_int_prim_variables...,
        tuples::get<
            Tags::detail::TemporaryReference<ValenciaInteriorEvolvedTags>>(
            shuffle_refs)...,
        valencia_int_dt_variables..., valencia_int_deriv_variables...,
        tuples::get<Tags::detail::TemporaryReference<
            ValenciaGridlessTags,
            tmpl::at<gridless_tags_and_types, ValenciaGridlessTags>>>(
            shuffle_refs)...);
    if (not gh_string.has_value()) {
      return valencia_string;
    }
    if (not valencia_string.has_value()) {
      return gh_string;
    }
    return gh_string.value() + ";" + valencia_string.value();
  }

  template <typename... GridlessVariables>
  static std::optional<std::string> dg_time_derivative(
      const DerivedGhCondition& gh_condition,
      const DerivedValenciaCondition& valencia_condition,

      const gsl::not_null<typename GhEvolvedTags::type*>... gh_dt_variables,
      const gsl::not_null<
          typename ValenciaEvolvedTags::type*>... valencia_dt_variables,

      const std::optional<tnsr::I<DataVector, 3_st, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3_st, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3_st, Frame::Inertial>& normal_vector,

      const typename DeduplicatedInteriorEvolvedTags::
          type&... int_evolved_variables,

      const typename GhInteriorPrimitiveTags::type&... gh_int_prim_variables,
      const typename ValenciaInteriorPrimitiveTags::
          type&... valencia_int_prim_variables,

      const typename DeduplicatedTempTags::type&... temp_variables,

      const typename GhInteriorDtTags::type&... gh_int_dt_variables,
      const typename ValenciaInteriorDtTags::type&... valencia_int_dt_variables,

      const typename GhInteriorDerivTags::type&... gh_int_deriv_variables,
      const typename ValenciaInteriorDerivTags::
          type&... valencia_int_deriv_variables,

      const GridlessVariables&... gridless_variables) noexcept {
    using gridless_tags_and_types =
        tmpl::map<tmpl::pair<DeduplicatedGridlessTags, GridlessVariables>...>;

    tuples::TaggedTuple<
        Tags::detail::TemporaryReference<
            DeduplicatedGridlessTags,
            tmpl::at<gridless_tags_and_types, DeduplicatedGridlessTags>>...,
        Tags::detail::TemporaryReference<DeduplicatedTempTags>...,
        Tags::detail::TemporaryReference<DeduplicatedInteriorEvolvedTags>...>
        shuffle_refs{gridless_variables..., temp_variables...,
                     int_evolved_variables...};

    std::optional<std::string> gh_string{};
    if constexpr (DerivedGhCondition::bc_type ==
                      evolution::BoundaryConditions::Type::TimeDerivative or
                  DerivedGhCondition::bc_type ==
                      evolution::BoundaryConditions::Type::
                          GhostAndTimeDerivative) {
      gh_string = gh_condition.dg_time_derivative(
          gh_dt_variables..., face_mesh_velocity, normal_covector,
          normal_vector,
          tuples::get<Tags::detail::TemporaryReference<GhInteriorEvolvedTags>>(
              shuffle_refs)...,
          gh_int_prim_variables...,
          tuples::get<Tags::detail::TemporaryReference<GhInteriorTempTags>>(
              shuffle_refs)...,
          gh_int_dt_variables..., gh_int_deriv_variables...,
          tuples::get<Tags::detail::TemporaryReference<
              GhGridlessTags,
              tmpl::at<gridless_tags_and_types, GhGridlessTags>>>(
              shuffle_refs)...);
    }
    std::optional<std::string> valencia_string{};
    if constexpr (DerivedValenciaCondition::bc_type ==
                      evolution::BoundaryConditions::Type::TimeDerivative or
                  DerivedValenciaCondition::bc_type ==
                      evolution::BoundaryConditions::Type::
                          GhostAndTimeDerivative) {
      valencia_string = valencia_condition.dg_time_derivative(
          valencia_dt_variables..., face_mesh_velocity, normal_covector,
          normal_vector,
          tuples::get<
              Tags::detail::TemporaryReference<ValenciaInteriorEvolvedTags>>(
              shuffle_refs)...,
          valencia_int_prim_variables...,
          tuples::get<
              Tags::detail::TemporaryReference<ValenciaInteriorTempTags>>(
              shuffle_refs)...,
          valencia_int_dt_variables..., valencia_int_deriv_variables...,
          tuples::get<Tags::detail::TemporaryReference<
              ValenciaGridlessTags,
              tmpl::at<gridless_tags_and_types, ValenciaGridlessTags>>>(
              shuffle_refs)...);
    }
    if (not gh_string.has_value()) {
      return valencia_string;
    }
    if (not valencia_string.has_value()) {
      return gh_string;
    }
    return gh_string.value() + ";" + valencia_string.value();
  }
};
}  // namespace detail

/*!
 * \brief Apply a boundary condition to the combined Generalized Harmonic (GH)
 * and Valencia GRMHD system using the boundary conditions defined separately
 * for the GH and Valencia systems.
 *
 * \details The implementation of this boundary condition applies the
 * `DerivedGhCondition` followed by the `DerivedValenciaCondition`.
 * It is anticipated that the systems are sufficiently independent that the
 * order of application is inconsequential. Arbitrary combinations of differing
 * `bc_type`s for the two systems are supported, with the only restriction that
 * if either is an outflow condition, both must be.
 */
template <typename DerivedGhCondition, typename DerivedValenciaCondition>
class ProductOfConditions final : public BoundaryCondition {
 public:
  static constexpr evolution::BoundaryConditions::Type bc_type =
      detail::UnionOfBcTypes<DerivedGhCondition::bc_type,
                             DerivedValenciaCondition::bc_type>::bc_type;

  using dg_interior_evolved_variables_tags =
      tmpl::remove_duplicates<tmpl::append<
          typename DerivedGhCondition::dg_interior_evolved_variables_tags,
          typename DerivedValenciaCondition::dg_interior_evolved_variables_tags,
          tmpl::conditional_t<
              DerivedGhCondition::bc_type ==
                      evolution::BoundaryConditions::Type::TimeDerivative and
                  bc_type == evolution::BoundaryConditions::Type::
                                 GhostAndTimeDerivative,
              typename detail::GhDoNothingGhostCondition::
                  dg_interior_evolved_variables_tags,
              tmpl::list<>>,
          tmpl::conditional_t<
              DerivedValenciaCondition::bc_type ==
                      evolution::BoundaryConditions::Type::TimeDerivative and
                  bc_type == evolution::BoundaryConditions::Type::
                                 GhostAndTimeDerivative,
              typename detail::ValenciaDoNothingGhostCondition::
                  dg_interior_evolved_variables_tags,
              tmpl::list<>>>>;
  using dg_interior_primitive_variables_tags = tmpl::append<
      tmpl::list<>,
      typename DerivedValenciaCondition::dg_interior_primitive_variables_tags>;
  using dg_interior_temporary_tags = tmpl::remove_duplicates<tmpl::append<
      typename DerivedGhCondition::dg_interior_temporary_tags,
      typename DerivedValenciaCondition::dg_interior_temporary_tags,
      tmpl::conditional_t<
          DerivedGhCondition::bc_type ==
                  evolution::BoundaryConditions::Type::TimeDerivative and
              bc_type ==
                  evolution::BoundaryConditions::Type::GhostAndTimeDerivative,
          typename detail::GhDoNothingGhostCondition::
              dg_interior_temporary_tags,
          tmpl::list<>>,
      tmpl::conditional_t<
          DerivedValenciaCondition::bc_type ==
                  evolution::BoundaryConditions::Type::TimeDerivative and
              bc_type ==
                  evolution::BoundaryConditions::Type::GhostAndTimeDerivative,
          typename detail::ValenciaDoNothingGhostCondition::
              dg_interior_temporary_tags,
          tmpl::list<>>>>;
  using dg_gridless_tags = tmpl::remove_duplicates<
      tmpl::append<typename DerivedGhCondition::dg_gridless_tags,
                   typename DerivedValenciaCondition::dg_gridless_tags>>;
  using dg_interior_dt_vars_tags = tmpl::append<
      evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          DerivedGhCondition>,
      evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          DerivedValenciaCondition>>;
  using dg_interior_deriv_vars_tags = tmpl::append<
      evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          DerivedGhCondition>,
      evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          DerivedValenciaCondition>>;

  using product_of_conditions_impl = detail::ProductOfConditionsImpl<
      DerivedGhCondition, DerivedValenciaCondition,
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list,
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list,
      db::wrap_tags_in<
          ::Tags::Flux,
          typename GeneralizedHarmonic::System<3_st>::flux_variables,
          tmpl::size_t<3_st>, Frame::Inertial>,
      db::wrap_tags_in<::Tags::Flux,
                       typename grmhd::ValenciaDivClean::System::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      typename DerivedGhCondition::dg_interior_evolved_variables_tags,
      typename DerivedValenciaCondition::dg_interior_evolved_variables_tags,
      dg_interior_evolved_variables_tags, tmpl::list<>,
      typename DerivedValenciaCondition::dg_interior_primitive_variables_tags,
      typename DerivedGhCondition::dg_interior_temporary_tags,
      typename DerivedValenciaCondition::dg_interior_temporary_tags,
      dg_interior_temporary_tags,
      evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          DerivedGhCondition>,
      evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          DerivedValenciaCondition>,
      evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          DerivedGhCondition>,
      evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          DerivedValenciaCondition>,
      typename DerivedGhCondition::dg_gridless_tags,
      typename DerivedValenciaCondition::dg_gridless_tags, dg_gridless_tags>;

  static std::string name() noexcept {
    return "Product" + Options::name<DerivedGhCondition>() + "And" +
           Options::name<DerivedValenciaCondition>();
  }

  struct GhCondition {
    using type = DerivedGhCondition;
    static std::string name() noexcept {
      return "GeneralizedHarmonic" + Options::name<DerivedGhCondition>();
    }
    static constexpr Options::String help{
        "The Generalized Harmonic part of the product boundary condition"};
  };
  struct ValenciaCondition {
    using type = DerivedValenciaCondition;
    static std::string name() noexcept {
      return "Valencia" + Options::name<DerivedValenciaCondition>();
    }
    static constexpr Options::String help{
        "The Valencia part of the product boundary condition"};
  };

  using options = tmpl::list<GhCondition, ValenciaCondition>;

  static constexpr Options::String help = {
      "Direct product of a GH and ValenciaDivClean GRMHD boundary conditions. "
      "See the documentation for the two individual boundary conditions for "
      "further details."};

  ProductOfConditions() = default;
  ProductOfConditions(DerivedGhCondition gh_condition,
                      DerivedValenciaCondition valencia_condition) noexcept
      : derived_gh_condition_{gh_condition},
        derived_valencia_condition_{valencia_condition} {}
  ProductOfConditions(const ProductOfConditions&) = default;
  ProductOfConditions& operator=(const ProductOfConditions&) = default;
  ProductOfConditions(ProductOfConditions&&) = default;
  ProductOfConditions& operator=(ProductOfConditions&&) = default;
  ~ProductOfConditions() override = default;

  /// \cond
  explicit ProductOfConditions(CkMigrateMessage* msg) noexcept
      : BoundaryCondition(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, ProductOfConditions);
  /// \endcond

  void pup(PUP::er& p) noexcept override;

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  template <typename... Args>
  std::optional<std::string> dg_ghost(Args&&... args) const noexcept {
    return product_of_conditions_impl::dg_ghost(derived_gh_condition_,
                                                derived_valencia_condition_,
                                                std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::optional<std::string> dg_outflow(Args&&... args) const noexcept {
    return product_of_conditions_impl::dg_outflow(derived_gh_condition_,
                                                  derived_valencia_condition_,
                                                  std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::optional<std::string> dg_time_derivative(Args&&... args) const noexcept {
    return product_of_conditions_impl::dg_time_derivative(
        derived_gh_condition_, derived_valencia_condition_,
        std::forward<Args>(args)...);
  }

 private:
  DerivedGhCondition derived_gh_condition_;
  DerivedValenciaCondition derived_valencia_condition_;
};

template <typename DerivedGhCondition, typename DerivedValenciaCondition>
void ProductOfConditions<DerivedGhCondition, DerivedValenciaCondition>::pup(
    PUP::er& p) noexcept {
  p | derived_gh_condition_;
  p | derived_valencia_condition_;
  BoundaryCondition::pup(p);
}

template <typename DerivedGhCondition, typename DerivedValenciaCondition>
auto ProductOfConditions<DerivedGhCondition,
                         DerivedValenciaCondition>::get_clone() const noexcept
    -> std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> {
  return std::make_unique<ProductOfConditions>(*this);
}

/// \cond
template <typename DerivedGhCondition, typename DerivedValenciaCondition>
PUP::able::PUP_ID ProductOfConditions<DerivedGhCondition,
                                      DerivedValenciaCondition>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
