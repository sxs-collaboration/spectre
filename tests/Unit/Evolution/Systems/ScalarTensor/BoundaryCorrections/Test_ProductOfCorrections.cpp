// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename DerivedCorrection, typename PackagedFieldTagList,
          typename EvolvedTagList, typename TempTagList, typename VolumeTagList>
struct ComputeBoundaryCorrectionHelperImpl;

template <typename DerivedCorrection, typename... PackagedFieldTags,
          typename... EvolvedTags, typename... TempTags, typename... VolumeTags>
struct ComputeBoundaryCorrectionHelperImpl<
    DerivedCorrection, tmpl::list<PackagedFieldTags...>,
    tmpl::list<EvolvedTags...>, tmpl::list<TempTags...>,
    tmpl::list<VolumeTags...>> {
  template <typename PackagedVariables, typename EvolvedVariables,
            typename TempVariables,
            typename VolumeVariables>
  static double dg_package_data(
      const gsl::not_null<PackagedVariables*> packaged_variables,
      const EvolvedVariables& evolved_variables,
      const TempVariables& temp_variables,
      const VolumeVariables& volume_variables,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const DerivedCorrection& derived_correction) {
    return derived_correction.dg_package_data(
        make_not_null(&get<PackagedFieldTags>(*packaged_variables))...,
        get<EvolvedTags>(evolved_variables)...,
        get<TempTags>(temp_variables)..., normal_covector, normal_vector,
        mesh_velocity, normal_dot_mesh_velocity,
        get<VolumeTags>(volume_variables)...);
  }

  template <typename EvolvedVariables, typename PackagedVariables>
  static void dg_boundary_terms(
      const gsl::not_null<EvolvedVariables*> boundary_corrections,
      const PackagedVariables& internal_packaged_fields,
      const PackagedVariables& external_packaged_fields,
      dg::Formulation dg_formulation,
      const DerivedCorrection& derived_correction) {
    derived_correction.dg_boundary_terms(
        make_not_null(&get<EvolvedTags>(*boundary_corrections))...,
        get<PackagedFieldTags>(internal_packaged_fields)...,
        get<PackagedFieldTags>(external_packaged_fields)..., dg_formulation);
  }
};

template <typename DerivedCorrection, typename EvolvedTagList
          >
using ComputeBoundaryCorrectionHelper = ComputeBoundaryCorrectionHelperImpl<
    DerivedCorrection, typename DerivedCorrection::dg_package_field_tags,
    EvolvedTagList,
    typename DerivedCorrection::dg_package_data_temporary_tags,
    typename DerivedCorrection::dg_package_data_volume_tags>;

template <typename DerivedGhCorrection, typename DerivedScalarCorrection>
void test_boundary_correction_combination(
    const DerivedGhCorrection& derived_gh_correction,
    const DerivedScalarCorrection& derived_scalar_correction,
    const ScalarTensor::BoundaryCorrections::ProductOfCorrections<
        DerivedGhCorrection, DerivedScalarCorrection>&
        derived_product_correction,
    const dg::Formulation formulation) {
  CHECK(derived_product_correction.gh_correction() == derived_gh_correction);
  CHECK(derived_product_correction.scalar_correction() ==
        derived_scalar_correction);
  using gh_variables_tags = typename gh::System<3>::variables_tag::tags_list;
  using scalar_variables_tags =
      typename CurvedScalarWave::System<3>::variables_tag::tags_list;
  using evolved_variables_type =
      Variables<tmpl::append<gh_variables_tags, scalar_variables_tags>>;

  using derived_product_correction_type =
      ScalarTensor::BoundaryCorrections::ProductOfCorrections<
          DerivedGhCorrection, DerivedScalarCorrection>;

  using packaged_variables_type = Variables<
      typename derived_product_correction_type::dg_package_field_tags>;

  using temporary_variables_type = Variables<
      typename derived_product_correction_type::dg_package_data_temporary_tags>;

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
      ComputeBoundaryCorrectionHelper<DerivedGhCorrection, gh_variables_tags>::
          dg_package_data(make_not_null(&expected_packaged_variables),
                          evolved_variables, temporary_variables,
                          volume_variables, normal_covector, normal_vector,
                          mesh_velocity, normal_dot_mesh_velocity,
                          derived_gh_correction);
  double expected_package_scalar_result =
      ComputeBoundaryCorrectionHelper<DerivedScalarCorrection,
                                      scalar_variables_tags>::
          dg_package_data(make_not_null(&expected_packaged_variables),
                          evolved_variables, temporary_variables,
                          volume_variables, normal_covector, normal_vector,
                          mesh_velocity, normal_dot_mesh_velocity,
                          derived_scalar_correction);

  double package_combined_result = ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, scalar_variables_tags>>::
      dg_package_data(make_not_null(&packaged_variables), evolved_variables,
                      temporary_variables, volume_variables, normal_covector,
                      normal_vector, mesh_velocity, normal_dot_mesh_velocity,
                      derived_product_correction);
  CHECK(approx(SINGLE_ARG(std::max(expected_package_gh_result,
                                   expected_package_scalar_result))) ==
        package_combined_result);
  CHECK_VARIABLES_APPROX(packaged_variables, expected_packaged_variables);

  auto serialized_and_deserialized_correction =
      serialize_and_deserialize(derived_product_correction);

  package_combined_result = ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, scalar_variables_tags>>::
      dg_package_data(make_not_null(&packaged_variables), evolved_variables,
                      temporary_variables, volume_variables, normal_covector,
                      normal_vector, mesh_velocity, normal_dot_mesh_velocity,
                      serialized_and_deserialized_correction);
  CHECK(approx(SINGLE_ARG(std::max(expected_package_gh_result,
                                   expected_package_scalar_result))) ==
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
  ComputeBoundaryCorrectionHelper<DerivedGhCorrection, gh_variables_tags>::
      dg_boundary_terms(make_not_null(&expected_boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, derived_gh_correction);
  ComputeBoundaryCorrectionHelper<DerivedScalarCorrection,
                                  scalar_variables_tags>::
      dg_boundary_terms(make_not_null(&expected_boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, derived_scalar_correction);

  ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, scalar_variables_tags>>::
      dg_boundary_terms(make_not_null(&boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, derived_product_correction);
  CHECK_VARIABLES_APPROX(boundary_correction, expected_boundary_correction);

  ComputeBoundaryCorrectionHelper<
      derived_product_correction_type,
      tmpl::append<gh_variables_tags, scalar_variables_tags>>::
      dg_boundary_terms(make_not_null(&boundary_correction),
                        internal_packaged_fields, external_packaged_fields,
                        formulation, serialized_and_deserialized_correction);
  CHECK_VARIABLES_APPROX(boundary_correction, expected_boundary_correction);

  PUPable_reg(
      SINGLE_ARG(ScalarTensor::BoundaryCorrections::ProductOfCorrections<
                 DerivedGhCorrection, DerivedScalarCorrection>));

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      ScalarTensor::System>(
      make_not_null(&gen), derived_product_correction,
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});
}
}  // namespace

SPECTRE_TEST_CASE(
 "Unit.Evolution.Systems.ScalarTensor.BoundaryCorrections.ProductOfCorrections",
    "[Unit][Evolution]") {
  {
    INFO("Product correction UpwindPenalty and UpwindPenalty");
    CurvedScalarWave::BoundaryCorrections::UpwindPenalty<3> scalar_correction{};
    gh::BoundaryCorrections::UpwindPenalty<3> gh_correction{};
    TestHelpers::test_creation<
        std::unique_ptr<ScalarTensor::BoundaryCorrections::BoundaryCorrection>>(
        "ProductUpwindPenaltyGHAndUpwindPenaltyScalar:\n"
        "  UpwindPenaltyGH:\n"
        "  UpwindPenaltyScalar:");
    ScalarTensor::BoundaryCorrections::ProductOfCorrections<
        gh::BoundaryCorrections::UpwindPenalty<3>,
        CurvedScalarWave::BoundaryCorrections::UpwindPenalty<3>>
        product_boundary_correction{gh_correction, scalar_correction};
    for (const auto formulation :
         {dg::Formulation::StrongInertial, dg::Formulation::WeakInertial}) {
      test_boundary_correction_combination<
          gh::BoundaryCorrections::UpwindPenalty<3>,
          CurvedScalarWave::BoundaryCorrections::UpwindPenalty<3>>(
          gh_correction, scalar_correction, product_boundary_correction,
          formulation);
    }
  }
}
