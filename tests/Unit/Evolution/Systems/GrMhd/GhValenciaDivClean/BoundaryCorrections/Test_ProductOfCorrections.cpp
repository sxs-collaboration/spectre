// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename DerivedCorrection, typename PackagedFieldTagList,
          typename EvolvedTagList, typename FluxTagList, typename TempTagList,
          typename PrimTagList, typename VolumeTagList>
struct ComputeBoundaryCorrectionHelperImpl;

template <typename DerivedCorrection, typename... PackagedFieldTags,
          typename... EvolvedTags, typename... FluxTags, typename... TempTags,
          typename... PrimTags, typename... VolumeTags>
struct ComputeBoundaryCorrectionHelperImpl<
    DerivedCorrection, tmpl::list<PackagedFieldTags...>,
    tmpl::list<EvolvedTags...>, tmpl::list<FluxTags...>,
    tmpl::list<TempTags...>, tmpl::list<PrimTags...>,
    tmpl::list<VolumeTags...>> {
  template <typename PackagedVariables, typename EvolvedVariables,
            typename FluxVariables, typename TempVariables,
            typename PrimVariables, typename VolumeVariables>
  static double dg_package_data(
      const gsl::not_null<PackagedVariables*> packaged_variables,
      const EvolvedVariables& evolved_variables,
      const FluxVariables& flux_variables, const TempVariables& temp_variables,
      const PrimVariables& prim_variables,
      const VolumeVariables& volume_variables,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const DerivedCorrection& derived_correction) noexcept {
    return derived_correction.dg_package_data(
        make_not_null(&get<PackagedFieldTags>(*packaged_variables))...,
        get<EvolvedTags>(evolved_variables)...,
        get<FluxTags>(flux_variables)..., get<TempTags>(temp_variables)...,
        get<PrimTags>(prim_variables)..., normal_covector, normal_vector,
        mesh_velocity, normal_dot_mesh_velocity,
        get<VolumeTags>(volume_variables)...);
  }

  template <typename EvolvedVariables, typename PackagedVariables>
  static void dg_boundary_terms(
      const gsl::not_null<EvolvedVariables*> boundary_corrections,
      const PackagedVariables& internal_packaged_fields,
      const PackagedVariables& external_packaged_fields,
      dg::Formulation dg_formulation,
      const DerivedCorrection& derived_correction) noexcept {
    derived_correction.dg_boundary_terms(
        make_not_null(&get<EvolvedTags>(*boundary_corrections))...,
        get<PackagedFieldTags>(internal_packaged_fields)...,
        get<PackagedFieldTags>(external_packaged_fields)..., dg_formulation);
  }
};

template <typename DerivedCorrection, typename EvolvedTagList,
          typename FluxTagList>
using ComputeBoundaryCorrectionHelper = ComputeBoundaryCorrectionHelperImpl<
    DerivedCorrection, typename DerivedCorrection::dg_package_field_tags,
    EvolvedTagList, FluxTagList,
    typename DerivedCorrection::dg_package_data_temporary_tags,
    typename DerivedCorrection::dg_package_data_primitive_tags,
    typename DerivedCorrection::dg_package_data_volume_tags>;

template <typename DerivedGhCorrection, typename DerivedValenciaCorrection>
void test_boundary_correction_combination(
    const DerivedGhCorrection& derived_gh_correction,
    const DerivedValenciaCorrection& derived_valencia_correction,
    const grmhd::GhValenciaDivClean::BoundaryCorrections::ProductOfCorrections<
        DerivedGhCorrection, DerivedValenciaCorrection>&
        derived_product_correction,
    const dg::Formulation formulation) noexcept {
  using gh_variables_tags =
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list;
  using valencia_variables_tags =
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list;
  using evolved_variables_type =
      Variables<tmpl::append<gh_variables_tags, valencia_variables_tags>>;

  using derived_product_correction_type =
      grmhd::GhValenciaDivClean::BoundaryCorrections::ProductOfCorrections<
          DerivedGhCorrection, DerivedValenciaCorrection>;

  using packaged_variables_type = Variables<
      typename derived_product_correction_type::dg_package_field_tags>;

  using gh_flux_tags = db::wrap_tags_in<
      ::Tags::Flux, typename GeneralizedHarmonic::System<3_st>::flux_variables,
      tmpl::size_t<3_st>, Frame::Inertial>;
  using valencia_flux_tags =
      db::wrap_tags_in<::Tags::Flux,
                       typename grmhd::ValenciaDivClean::System::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>;
  using flux_variables_type =
      Variables<tmpl::append<gh_flux_tags, valencia_flux_tags>>;

  using temporary_variables_type = Variables<
      typename derived_product_correction_type::dg_package_data_temporary_tags>;
  using primitive_variables_type = Variables<
      typename derived_product_correction_type::dg_package_data_primitive_tags>;
  using volume_variables_type = Variables<
      typename derived_product_correction_type::dg_package_data_volume_tags>;

  const size_t element_size = 10_st;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 1.0);

  packaged_variables_type expected_packaged_variables{element_size};
  packaged_variables_type packaged_variables{element_size};

  const auto evolved_variables =
      make_with_random_values<evolved_variables_type>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  const auto flux_variables = make_with_random_values<flux_variables_type>(
      make_not_null(&gen), make_not_null(&dist), element_size);
  primitive_variables_type prim_variables{element_size};
  if constexpr (tmpl::size<typename derived_product_correction_type::
                               dg_package_data_primitive_tags>::value > 0) {
    fill_with_random_values(make_not_null(&prim_variables), make_not_null(&gen),
                            make_not_null(&dist));
  }
  const auto temporary_variables =
      make_with_random_values<temporary_variables_type>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  volume_variables_type volume_variables{element_size};
  if constexpr (tmpl::size<typename derived_product_correction_type::
                               dg_package_data_volume_tags>::value > 0) {
    fill_with_random_values(make_not_null(&volume_variables),
                            make_not_null(&gen), make_not_null(&dist));
  }

  const auto normal_covector =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  const auto normal_vector =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  const auto mesh_velocity =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  const auto normal_dot_mesh_velocity =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&gen), make_not_null(&dist), element_size);

  double expected_package_gh_result =
      ComputeBoundaryCorrectionHelper<DerivedGhCorrection, gh_variables_tags,
                                      gh_flux_tags>::
          dg_package_data(make_not_null(&expected_packaged_variables),
                          evolved_variables, flux_variables,
                          temporary_variables, prim_variables, volume_variables,
                          normal_covector, normal_vector, mesh_velocity,
                          normal_dot_mesh_velocity, derived_gh_correction);
  double expected_package_valencia_result = ComputeBoundaryCorrectionHelper<
      DerivedValenciaCorrection, valencia_variables_tags, valencia_flux_tags>::
      dg_package_data(make_not_null(&expected_packaged_variables),
                      evolved_variables, flux_variables, temporary_variables,
                      prim_variables, volume_variables, normal_covector,
                      normal_vector, mesh_velocity, normal_dot_mesh_velocity,
                      derived_valencia_correction);

  double package_combined_result = ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, valencia_variables_tags>,
      tmpl::append<gh_flux_tags, valencia_flux_tags>>::
      dg_package_data(make_not_null(&packaged_variables), evolved_variables,
                      flux_variables, temporary_variables, prim_variables,
                      volume_variables, normal_covector, normal_vector,
                      mesh_velocity, normal_dot_mesh_velocity,
                      derived_product_correction);
  CHECK(approx(SINGLE_ARG(std::max(expected_package_gh_result,
                                   expected_package_valencia_result))) ==
        package_combined_result);
  CHECK_VARIABLES_APPROX(packaged_variables, expected_packaged_variables);

  auto serialized_and_deserialized_correction =
      serialize_and_deserialize(derived_product_correction);

  package_combined_result = ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, valencia_variables_tags>,
      tmpl::append<gh_flux_tags, valencia_flux_tags>>::
      dg_package_data(make_not_null(&packaged_variables), evolved_variables,
                      flux_variables, temporary_variables, prim_variables,
                      volume_variables, normal_covector, normal_vector,
                      mesh_velocity, normal_dot_mesh_velocity,
                      serialized_and_deserialized_correction);
  CHECK(approx(SINGLE_ARG(std::max(expected_package_gh_result,
                                   expected_package_valencia_result))) ==
        package_combined_result);
  CHECK_VARIABLES_APPROX(packaged_variables, expected_packaged_variables);

  const auto external_packaged_fields =
      make_with_random_values<packaged_variables_type>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  const auto internal_packaged_fields =
      make_with_random_values<packaged_variables_type>(
          make_not_null(&gen), make_not_null(&dist), element_size);

  evolved_variables_type expected_boundary_correction{element_size};
  evolved_variables_type boundary_correction{element_size};
  ComputeBoundaryCorrectionHelper<DerivedGhCorrection, gh_variables_tags,
                                  gh_flux_tags>::
      dg_boundary_terms(make_not_null(&expected_boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, derived_gh_correction);
  ComputeBoundaryCorrectionHelper<DerivedValenciaCorrection,
                                  valencia_variables_tags, valencia_flux_tags>::
      dg_boundary_terms(make_not_null(&expected_boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, derived_valencia_correction);

  ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, valencia_variables_tags>,
      tmpl::append<gh_flux_tags, valencia_flux_tags>>::
      dg_boundary_terms(make_not_null(&boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, derived_product_correction);
  CHECK_VARIABLES_APPROX(boundary_correction, expected_boundary_correction);

  ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, valencia_variables_tags>,
      tmpl::append<gh_flux_tags, valencia_flux_tags>>::
      dg_boundary_terms(make_not_null(&boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, serialized_and_deserialized_correction);
  CHECK_VARIABLES_APPROX(boundary_correction, expected_boundary_correction);

  PUPable_reg(SINGLE_ARG(
      grmhd::GhValenciaDivClean::BoundaryCorrections::ProductOfCorrections<
          DerivedGhCorrection, DerivedValenciaCorrection>));

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      grmhd::GhValenciaDivClean::System>(
      make_not_null(&gen), derived_product_correction,
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GhValenciaDivClean.BoundaryCorrections.ProductOfCorrections",
    "[Unit][Evolution]") {
  // scoped to separate out each product combination
  {
    INFO("Product correction UpwindPenalty and Rusanov");
    grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov valencia_correction{};
    GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<3_st>
        gh_correction{};
    TestHelpers::test_creation<std::unique_ptr<
        grmhd::GhValenciaDivClean::BoundaryCorrections::BoundaryCorrection>>(
        "ProductUpwindPenaltyAndRusanov:\n"
        "  UpwindPenalty:\n"
        "  Rusanov:");
    grmhd::GhValenciaDivClean::BoundaryCorrections::ProductOfCorrections<
        GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<3_st>,
        grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov>
        product_boundary_correction{gh_correction, valencia_correction};
    for (const auto formulation :
         {dg::Formulation::StrongInertial, dg::Formulation::WeakInertial}) {
      test_boundary_correction_combination<
          GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<3_st>,
          grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov>(
          gh_correction, valencia_correction, product_boundary_correction,
          formulation);
    }
  }
}
