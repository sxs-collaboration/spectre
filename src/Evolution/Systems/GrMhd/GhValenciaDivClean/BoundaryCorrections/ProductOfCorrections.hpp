// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryCorrections {
namespace detail {
template <typename DerivedGhCorrection, typename DerivedValenciaCorrection,
          typename GhPackageFieldTagList, typename ValenciaPackageFieldTagList,
          typename GhEvolvedTagList, typename ValenciaEvolvedTags,
          typename GhFluxTagList, typename ValenciaFluxTagList,
          typename GhTempTagList, typename ValenciaTempTagList,
          typename DeduplicatedTempTags, typename GhPrimTagList,
          typename ValenicaPrimTagList, typename GhVolumeTagList,
          typename ValenciaVolumeTagList>
struct ProductOfCorrectionsImpl;

template <typename DerivedGhCorrection, typename DerivedValenciaCorrection,
          typename... GhPackageFieldTags, typename... ValenciaPackageFieldTags,
          typename... GhEvolvedTags, typename... ValenciaEvolvedTags,
          typename... GhFluxTags, typename... ValenciaFluxTags,
          typename... GhTempTags, typename... ValenciaTempTags,
          typename... DeduplicatedTempTags, typename... GhPrimTags,
          typename... ValenciaPrimTags, typename... GhVolumeTags,
          typename... ValenciaVolumeTags>
struct ProductOfCorrectionsImpl<
    DerivedGhCorrection, DerivedValenciaCorrection,
    tmpl::list<GhPackageFieldTags...>, tmpl::list<ValenciaPackageFieldTags...>,
    tmpl::list<GhEvolvedTags...>, tmpl::list<ValenciaEvolvedTags...>,
    tmpl::list<GhFluxTags...>, tmpl::list<ValenciaFluxTags...>,
    tmpl::list<GhTempTags...>, tmpl::list<ValenciaTempTags...>,
    tmpl::list<DeduplicatedTempTags...>, tmpl::list<GhPrimTags...>,
    tmpl::list<ValenciaPrimTags...>, tmpl::list<GhVolumeTags...>,
    tmpl::list<ValenciaVolumeTags...>> {
  static double dg_package_data(
      const gsl::not_null<
          typename GhPackageFieldTags::type*>... gh_packaged_fields,
      const gsl::not_null<
          typename ValenciaPackageFieldTags::type*>... valencia_packaged_fields,

      const typename GhEvolvedTags::type&... gh_variables,
      const typename ValenciaEvolvedTags::type&... valencia_variables,

      const typename GhFluxTags::type&... gh_fluxes,
      const typename ValenciaFluxTags::type&... valencia_fluxes,

      const typename DeduplicatedTempTags::type&... temporaries,

      const typename GhPrimTags::type&... gh_primitives,
      const typename ValenciaPrimTags::type&... valencia_primitives,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const typename GhVolumeTags::type&... gh_volume_quantities,
      const typename ValenciaVolumeTags::type&... valencia_volume_quantities,

      const DerivedGhCorrection& gh_correction,
      const DerivedValenciaCorrection& valencia_correction) noexcept {
    tuples::TaggedTuple<
        Tags::detail::TemporaryReference<DeduplicatedTempTags>...>
        shuffle_refs{temporaries...};
    return std::max(
        gh_correction.dg_package_data(
            gh_packaged_fields..., gh_variables..., gh_fluxes...,
            tuples::get<Tags::detail::TemporaryReference<GhTempTags>>(
                shuffle_refs)...,
            gh_primitives..., normal_covector, normal_vector, mesh_velocity,
            normal_dot_mesh_velocity, gh_volume_quantities...),
        valencia_correction.dg_package_data(
            valencia_packaged_fields..., valencia_variables...,
            valencia_fluxes...,
            tuples::get<Tags::detail::TemporaryReference<ValenciaTempTags>>(
                shuffle_refs)...,
            valencia_primitives..., normal_covector, normal_vector,
            mesh_velocity, normal_dot_mesh_velocity,
            valencia_volume_quantities...));
  }

  static void dg_boundary_terms(
      const gsl::not_null<
          typename GhEvolvedTags::type*>... gh_boundary_corrections,
      const gsl::not_null<
          typename ValenciaEvolvedTags::type*>... valencia_boundary_corrections,

      const typename GhPackageFieldTags::type&... gh_internal_packaged_fields,
      const typename ValenciaPackageFieldTags::
          type&... valencia_internal_packaged_fields,

      const typename GhPackageFieldTags::type&... gh_external_packaged_fields,
      const typename ValenciaPackageFieldTags::
          type&... valencia_external_packaged_fields,
      const dg::Formulation dg_formulation,

      const DerivedGhCorrection& gh_correction,
      const DerivedValenciaCorrection& valencia_correction) noexcept {
    gh_correction.dg_boundary_terms(
        gh_boundary_corrections..., gh_internal_packaged_fields...,
        gh_external_packaged_fields..., dg_formulation);
    valencia_correction.dg_boundary_terms(
        valencia_boundary_corrections..., valencia_internal_packaged_fields...,
        valencia_external_packaged_fields..., dg_formulation);
  }
};
}  // namespace detail

/*!
 * \brief Apply a boundary condition to the combined Generalized Harmonic (GH)
 * and Valencia GRMHD system using boundary corrections defined separately for
 * the GH and Valencia systems.
 *
 * \details The implementation of this boundary correction applies the
 * `DerivedGhCorrection` followed by the `DerivedValenciaCorrection`. It is
 * anticipated that the systems are sufficiently independent that the order of
 * application is inconsequential.
 */
template <typename DerivedGhCorrection, typename DerivedValenciaCorrection>
class ProductOfCorrections : public BoundaryCorrection {
 public:
  using dg_package_field_tags =
      tmpl::append<typename DerivedGhCorrection::dg_package_field_tags,
                   typename DerivedValenciaCorrection::dg_package_field_tags>;

  using dg_package_data_temporary_tags = tmpl::remove_duplicates<tmpl::append<
      typename DerivedGhCorrection::dg_package_data_temporary_tags,
      typename DerivedValenciaCorrection::dg_package_data_temporary_tags>>;

  using dg_package_data_primitive_tags =
      typename DerivedValenciaCorrection::dg_package_data_primitive_tags;

  using dg_package_data_volume_tags = tmpl::append<
      typename DerivedGhCorrection::dg_package_data_volume_tags,
      typename DerivedValenciaCorrection::dg_package_data_volume_tags>;

  using derived_product_correction_impl = detail::ProductOfCorrectionsImpl<
      DerivedGhCorrection, DerivedValenciaCorrection,
      typename DerivedGhCorrection::dg_package_field_tags,
      typename DerivedValenciaCorrection::dg_package_field_tags,
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list,
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list,
      db::wrap_tags_in<
          ::Tags::Flux,
          typename GeneralizedHarmonic::System<3_st>::flux_variables,
          tmpl::size_t<3_st>, Frame::Inertial>,
      db::wrap_tags_in<::Tags::Flux,
                       typename grmhd::ValenciaDivClean::System::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      typename DerivedGhCorrection::dg_package_data_temporary_tags,
      typename DerivedValenciaCorrection::dg_package_data_temporary_tags,
      dg_package_data_temporary_tags, tmpl::list<>,
      typename DerivedValenciaCorrection::dg_package_data_primitive_tags,
      typename DerivedGhCorrection::dg_package_data_volume_tags,
      typename DerivedValenciaCorrection::dg_package_data_volume_tags>;

  static std::string name() noexcept {
    return "Product" + Options::name<DerivedGhCorrection>() + "And" +
           Options::name<DerivedValenciaCorrection>();
  }

  struct GhCorrection {
    using type = DerivedGhCorrection;
    static std::string name() noexcept {
      return Options::name<DerivedGhCorrection>();
    }
    static constexpr Options::String help{
        "The Generalized Harmonic part of the product boundary condition"};
  };
  struct ValenciaCorrection {
    using type = DerivedValenciaCorrection;
    static std::string name() noexcept {
      return Options::name<DerivedValenciaCorrection>();
    }
    static constexpr Options::String help{
        "The Valencia part of the product boundary condition"};
  };

  using options = tmpl::list<GhCorrection, ValenciaCorrection>;

  static constexpr Options::String help = {
      "Direct product of a GH and ValenciaDivClean GRMHD boundary correction. "
      "See the documentation for the two individual boundary corrections for "
      "further details."};

  ProductOfCorrections() = default;
  ProductOfCorrections(DerivedGhCorrection gh_correction,
                       DerivedValenciaCorrection valencia_correction) noexcept
      : derived_gh_correction_{gh_correction},
        derived_valencia_correction_{valencia_correction} {}
  ProductOfCorrections(const ProductOfCorrections&) = default;
  ProductOfCorrections& operator=(const ProductOfCorrections&) = default;
  ProductOfCorrections(ProductOfCorrections&&) = default;
  ProductOfCorrections& operator=(ProductOfCorrections&&) = default;
  ~ProductOfCorrections() override = default;

  /// \cond
  explicit ProductOfCorrections(CkMigrateMessage* msg) noexcept
      : BoundaryCorrection(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ProductOfCorrections);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) noexcept override {
    p | derived_gh_correction_;
    p | derived_valencia_correction_;
    BoundaryCorrection::pup(p);
  }

  std::unique_ptr<BoundaryCorrection> get_clone() const noexcept override {
    return std::make_unique<ProductOfCorrections>(*this);
  }

  template <typename... Args>
  double dg_package_data(Args&&... args) const noexcept {
    return derived_product_correction_impl::dg_package_data(
        std::forward<Args>(args)..., derived_gh_correction_,
        derived_valencia_correction_);
  }

  template <typename... Args>
  void dg_boundary_terms(Args&&... args) const noexcept {
    derived_product_correction_impl::dg_boundary_terms(
        std::forward<Args>(args)..., derived_gh_correction_,
        derived_valencia_correction_);
  }

 private:
  DerivedGhCorrection derived_gh_correction_;
  DerivedValenciaCorrection derived_valencia_correction_;
};

/// \cond
template <typename DerivedGhCorrection, typename DerivedValenciaCorrection>
PUP::able::PUP_ID ProductOfCorrections<DerivedGhCorrection,
                                       DerivedValenciaCorrection>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace grmhd::GhValenciaDivClean::BoundaryCorrections
