// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/ProductOfConditions.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/KerrSphericalHarmonic.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct DummyAnalyticSolutionTag : db::SimpleTag, Tags::AnalyticSolutionOrData {
  using type = gh::Solutions::WrappedGr<
      ScalarTensor::AnalyticData::KerrSphericalHarmonic>;
};

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            ScalarTensor::BoundaryConditions::BoundaryCondition,
            tmpl::list<ScalarTensor::BoundaryConditions::ProductOfConditions<
                gh::BoundaryConditions::DirichletAnalytic<3_st>,
                CurvedScalarWave::BoundaryConditions::AnalyticConstant<3_st>>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<gh::Solutions::WrappedGr<
                       ScalarTensor::AnalyticData::KerrSphericalHarmonic>>>>;
  };
};

template <typename DerivedCondition, typename EvolvedTagList,
          typename MutatedTagList, typename ArgumentTagList,
          typename GridlessTagList>
struct ComputeBoundaryConditionHelper;

template <typename DerivedCondition, typename... EvolvedTags,
          typename... MutatedTags, typename... ArgumentTags,
          typename... GridlessTags>
struct ComputeBoundaryConditionHelper<
    DerivedCondition, tmpl::list<EvolvedTags...>, tmpl::list<MutatedTags...>,
    tmpl::list<ArgumentTags...>, tmpl::list<GridlessTags...>> {
  template <typename MutatedVariables, typename ArgumentVariables,
            typename GridlessBox>
  static std::optional<std::string> dg_ghost(
      const gsl::not_null<MutatedVariables*> mutated_variables,
      const ArgumentVariables& argument_variables,
      const GridlessBox& gridless_box,
      const DerivedCondition& derived_condition) {
    return derived_condition.dg_ghost(
        make_not_null(&get<MutatedTags>(*mutated_variables))...,
        tuples::get<ArgumentTags>(argument_variables)...,
        db::get<GridlessTags>(gridless_box)...);
  }

  template <typename ArgumentVariables, typename GridlessBox>
  static std::optional<std::string> dg_demand_outgoing_char_speeds(
      const ArgumentVariables& argument_variables,
      const GridlessBox& gridless_box,
      const DerivedCondition& derived_condition) {
    return derived_condition.dg_demand_outgoing_char_speeds(
        tuples::get<ArgumentTags>(argument_variables)...,
        db::get<GridlessTags>(gridless_box)...);
  }
};

template <typename GhCorrectionTempTagList,
          typename ScalarCorrectionTempTagList, typename DerivedGhCondition,
          typename DerivedScalarCondition, typename GridlessBox>
void test_boundary_condition_combination(
    const DerivedGhCondition& derived_gh_condition,
    const DerivedScalarCondition& derived_scalar_condition,
    const ScalarTensor::BoundaryConditions::ProductOfConditions<
        DerivedGhCondition, DerivedScalarCondition>& derived_product_condition,
    const GridlessBox& gridless_box) {
  using product_condition_type =
      ScalarTensor::BoundaryConditions::ProductOfConditions<
          DerivedGhCondition, DerivedScalarCondition>;

  using gh_mutable_tags = tmpl::append<
      typename gh::System<3_st>::variables_tag::tags_list,
      db::wrap_tags_in<::Tags::Flux, typename gh::System<3_st>::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      GhCorrectionTempTagList,
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3_st>>>;
  using scalar_mutable_tags = tmpl::append<
      typename CurvedScalarWave::System<3_st>::variables_tag::tags_list,
      db::wrap_tags_in<::Tags::Flux,
                       typename CurvedScalarWave::System<3_st>::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      ScalarCorrectionTempTagList,
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3_st>>>;
  using product_mutable_tags = tmpl::append<
      typename gh::System<3_st>::variables_tag::tags_list,
      typename CurvedScalarWave::System<3_st>::variables_tag::tags_list,
      db::wrap_tags_in<::Tags::Flux, typename gh::System<3_st>::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      db::wrap_tags_in<::Tags::Flux,
                       typename CurvedScalarWave::System<3_st>::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      tmpl::remove_duplicates<
          tmpl::append<GhCorrectionTempTagList, ScalarCorrectionTempTagList>>,
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3_st>>>;

  using gh_arg_tags = tmpl::append<
      tmpl::list<::domain::Tags::MeshVelocity<3_st>,
                 evolution::dg::Tags::NormalCovector<3_st>,
                 evolution::dg::Actions::detail::NormalVector<3_st>>,
      typename DerivedGhCondition::dg_interior_evolved_variables_tags,
      tmpl::list<>, typename DerivedGhCondition::dg_interior_temporary_tags,
      evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          DerivedGhCondition>,
      evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          DerivedGhCondition>>;
  using scalar_arg_tags = tmpl::append<
      tmpl::list<::domain::Tags::MeshVelocity<3_st>,
                 evolution::dg::Tags::NormalCovector<3_st>,
                 evolution::dg::Actions::detail::NormalVector<3_st>>,
      typename DerivedScalarCondition::dg_interior_evolved_variables_tags,
      typename DerivedScalarCondition::dg_interior_temporary_tags,
      evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          DerivedScalarCondition>,
      evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          DerivedScalarCondition>>;
  using product_arg_tags = tmpl::append<
      tmpl::list<::domain::Tags::MeshVelocity<3_st>,
                 evolution::dg::Tags::NormalCovector<3_st>,
                 evolution::dg::Actions::detail::NormalVector<3_st>>,
      typename product_condition_type::dg_interior_evolved_variables_tags,
      typename product_condition_type::dg_interior_temporary_tags,
      typename product_condition_type::dg_interior_dt_vars_tags,
      typename product_condition_type::dg_interior_deriv_vars_tags>;

  using gh_gridless_tags = typename DerivedGhCondition::dg_gridless_tags;
  using scalar_gridless_tags =
      typename DerivedScalarCondition::dg_gridless_tags;
  using product_gridless_tags =
      typename product_condition_type::dg_gridless_tags;

  const size_t element_size = 10_st;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 1.0);
  std::uniform_int_distribution<int> optional_dist(0, 1);

  auto mutable_variables =
      make_with_random_values<Variables<product_mutable_tags>>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  auto expected_mutable_variables =
      make_with_random_values<Variables<product_mutable_tags>>(
          make_not_null(&gen), make_not_null(&dist), element_size);
  tuples::tagged_tuple_from_typelist<product_arg_tags> argument_variables;
  tmpl::for_each<product_arg_tags>([&dist, &optional_dist, &gen, &element_size,
                                    &argument_variables](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    if constexpr (tt::is_a_v<std::optional, typename tag::type>) {
      if (optional_dist(gen) == 1) {
        tuples::get<tag>(argument_variables) =
            make_with_random_values<typename tag::type::value_type>(
                make_not_null(&gen), make_not_null(&dist), element_size);
      } else {
        tuples::get<tag>(argument_variables) = std::nullopt;
      }
    } else {
      tuples::get<tag>(argument_variables) =
          make_with_random_values<typename tag::type>(
              make_not_null(&gen), make_not_null(&dist), element_size);
    }
  });
  if constexpr (tmpl::list_contains_v<
                    product_arg_tags,
                    gr::Tags::SpacetimeMetric<DataVector, 3_st>>) {
    get<0, 0>(tuples::get<gr::Tags::SpacetimeMetric<DataVector, 3_st>>(
        argument_variables)) += -2.0;
    for (size_t i = 0; i < 3; ++i) {
      tuples::get<gr::Tags::SpacetimeMetric<DataVector, 3_st>>(
          argument_variables)
          .get(i + 1, 0) *= 0.01;
    }
  }

  using gh_bc_helper = ComputeBoundaryConditionHelper<
      DerivedGhCondition, typename gh::System<3_st>::variables_tag::tags_list,
      gh_mutable_tags, gh_arg_tags, gh_gridless_tags>;
  using scalar_bc_helper = ComputeBoundaryConditionHelper<
      DerivedScalarCondition,
      typename CurvedScalarWave::System<3_st>::variables_tag::tags_list,
      scalar_mutable_tags, scalar_arg_tags, scalar_gridless_tags>;
  using product_bc_helper = ComputeBoundaryConditionHelper<
      product_condition_type,
      tmpl::append<
          typename gh::System<3_st>::variables_tag::tags_list,
          typename CurvedScalarWave::System<3_st>::variables_tag::tags_list>,
      product_mutable_tags, product_arg_tags, product_gridless_tags>;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      product_condition_type::bc_type;

  if constexpr (bc_type ==
                evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds) {
    auto gh_result = gh_bc_helper::dg_demand_outgoing_char_speeds(
        argument_variables, gridless_box, derived_gh_condition);
    auto scalar_result = scalar_bc_helper::dg_demand_outgoing_char_speeds(
        argument_variables, gridless_box, derived_scalar_condition);
    auto product_result = product_bc_helper::dg_demand_outgoing_char_speeds(
        argument_variables, gridless_box, derived_product_condition);
    if (gh_result.has_value()) {
      if (scalar_result.has_value()) {
        CHECK((product_result.value() ==
               gh_result.value() + ";" + scalar_result.value()));
      } else {
        CHECK(product_result == gh_result);
      }
    } else {
      CHECK(product_result == scalar_result);
    }
    return;
  }
  std::optional<std::string> gh_result_ghost;
  std::optional<std::string> scalar_result_ghost;
  std::optional<std::string> product_result_ghost;

  std::optional<std::string> gh_result_time_derivative;
  std::optional<std::string> scalar_result_time_derivative;
  std::optional<std::string> product_result_time_derivative;

  if constexpr (DerivedGhCondition::bc_type ==
                evolution::BoundaryConditions::Type::Ghost) {
    gh_result_ghost = gh_bc_helper::dg_ghost(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_gh_condition);
  }

  if constexpr (DerivedScalarCondition::bc_type ==
                evolution::BoundaryConditions::Type::Ghost) {
    scalar_result_ghost = scalar_bc_helper::dg_ghost(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_scalar_condition);
  }

  if constexpr (bc_type == evolution::BoundaryConditions::Type::Ghost) {
    product_result_ghost = product_bc_helper::dg_ghost(
        make_not_null(&mutable_variables), argument_variables, gridless_box,
        derived_product_condition);
  }

  if (gh_result_ghost.has_value()) {
    if (scalar_result_ghost.has_value()) {
      CHECK((product_result_ghost.value() ==
             gh_result_ghost.value() + ";" + scalar_result_ghost.value()));
    } else {
      CHECK(product_result_ghost == gh_result_ghost);
    }
  } else {
    CHECK(product_result_ghost == scalar_result_ghost);
  }

  CHECK_VARIABLES_APPROX(mutable_variables, expected_mutable_variables);
}
}  // namespace

SPECTRE_TEST_CASE(
   "Unit.Evolution.Systems.ScalarTensor.BoundaryConditions.ProductOfConditions",
   "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  // scoped to separate out each product combination
  {
    INFO(
        "Product condition of DirichletAnalytic for Generalized Harmonic and "
        "AnalyticConstant for CurvedScalarWave.");

    const gh::BoundaryConditions::DirichletAnalytic<3_st> gh_condition{
        std::unique_ptr<evolution::initial_data::InitialData>(
            std::make_unique<gh::Solutions::WrappedGr<
                ScalarTensor::AnalyticData::KerrSphericalHarmonic>>(
                // Black Hole parameters
                1.5, std::array<double, 3>{{0.1, -0.2, 0.3}},
                // Scalar wave parameters
                2.3, 1.7, 2.9, std::pair<size_t, int>{1, 0}))};
    // Note: We test for non-zero amplitude of the scalar at the boundary for
    // robustness, however, this test initial data asymptotes to zero.
    const CurvedScalarWave::BoundaryConditions::AnalyticConstant<3_st>
        scalar_condition{// Amplitude
                         1.1};
    const auto product_boundary_condition = TestHelpers::test_creation<
        std::unique_ptr<ScalarTensor::BoundaryConditions::BoundaryCondition>,
        Metavariables>(
        "ProductDirichletAnalyticAndAnalyticConstant:\n"
        "  GeneralizedHarmonicDirichletAnalytic:\n"
        "    AnalyticPrescription:\n"
        "      KerrSphericalHarmonic:\n"
        "        Mass: 1.5\n"
        "        Spin: [0.1, -0.2, 0.3]\n"
        "        Amplitude: 2.3\n"
        "        Radius: 1.7\n"
        "        Width: 2.9\n"
        "        Mode: [1, 0]\n"
        "  ScalarAnalyticConstant:\n"
        "    Amplitude: 1.1\n");
    const auto gridless_box =
        db::create<db::AddSimpleTags<::Tags::Time, DummyAnalyticSolutionTag>>(
            0.5, gh::Solutions::WrappedGr<
                     ScalarTensor::AnalyticData::KerrSphericalHarmonic>{
                     // Black Hole parameters
                     1.5, std::array<double, 3>{{0.1, -0.2, 0.3}},
                     // Scalar wave parameters
                     2.3, 1.7, 2.9, std::pair<size_t, int>{1, 0}});
    auto serialized_and_deserialized_condition = serialize_and_deserialize(
        *dynamic_cast<ScalarTensor::BoundaryConditions::ProductOfConditions<
            gh::BoundaryConditions::DirichletAnalytic<3_st>,
            CurvedScalarWave::BoundaryConditions::AnalyticConstant<3_st>>*>(
            product_boundary_condition.get()));
    test_boundary_condition_combination<
        gh::BoundaryCorrections::UpwindPenalty<
            3_st>::dg_package_data_temporary_tags,
        CurvedScalarWave::BoundaryCorrections::UpwindPenalty<
            3_st>::dg_package_data_temporary_tags,
        gh::BoundaryConditions::DirichletAnalytic<3_st>,
        CurvedScalarWave::BoundaryConditions::AnalyticConstant<3_st>>(
        gh_condition, scalar_condition, serialized_and_deserialized_condition,
        gridless_box);
  }
  {
    INFO(
        "Product condition of CurvedScalarWave DemandOutgoingCharSpeeds and "
        "GeneralizedHarmonic DemandOutgoingCharSpeeds");
    const CurvedScalarWave::BoundaryConditions::DemandOutgoingCharSpeeds<3_st>
        scalar_condition{};
    const gh::BoundaryConditions::DemandOutgoingCharSpeeds<3_st> gh_condition{};
    const auto product_boundary_condition = TestHelpers::test_factory_creation<
        ScalarTensor::BoundaryConditions::BoundaryCondition,
        ScalarTensor::BoundaryConditions::ProductOfConditions<
            gh::BoundaryConditions::DemandOutgoingCharSpeeds<3_st>,
            CurvedScalarWave::BoundaryConditions::DemandOutgoingCharSpeeds<
                3_st>>>(
        "ProductDemandOutgoingCharSpeedsAndDemandOutgoingCharSpeeds:\n"
        "  GeneralizedHarmonicDemandOutgoingCharSpeeds:\n"
        "  ScalarDemandOutgoingCharSpeeds:");
    const auto gridless_box = db::create<db::AddSimpleTags<>>();
    auto serialized_and_deserialized_condition = serialize_and_deserialize(
        *dynamic_cast<ScalarTensor::BoundaryConditions::ProductOfConditions<
            gh::BoundaryConditions::DemandOutgoingCharSpeeds<3_st>,
            CurvedScalarWave::BoundaryConditions::DemandOutgoingCharSpeeds<
                3_st>>*>(product_boundary_condition.get()));
    test_boundary_condition_combination<
        tmpl::list<>, tmpl::list<>,
        gh::BoundaryConditions::DemandOutgoingCharSpeeds<3_st>,
        CurvedScalarWave::BoundaryConditions::DemandOutgoingCharSpeeds<3_st>>(
        gh_condition, scalar_condition, serialized_and_deserialized_condition,
        gridless_box);
  }
}
