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
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/ProductOfConditions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct DummyAnalyticSolutionTag : db::SimpleTag, Tags::AnalyticSolutionOrData {
  using type =
      GeneralizedHarmonic::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>;
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
      const DerivedCondition& derived_condition) noexcept {
    return derived_condition.dg_ghost(
        make_not_null(&get<MutatedTags>(*mutated_variables))...,
        tuples::get<ArgumentTags>(argument_variables)...,
        db::get<GridlessTags>(gridless_box)...);
  }

  template <typename ArgumentVariables, typename GridlessBox>
  static std::optional<std::string> dg_outflow(
      const ArgumentVariables& argument_variables,
      const GridlessBox& gridless_box,
      const DerivedCondition& derived_condition) noexcept {
    return derived_condition.dg_ghost(
        tuples::get<ArgumentTags>(argument_variables)...,
        db::get<GridlessTags>(gridless_box)...);
  }

  template <typename MutatedVariables, typename ArgumentVariables,
            typename GridlessBox>
  static std::optional<std::string> dg_time_derivative(
      const gsl::not_null<MutatedVariables*> mutated_variables,
      const ArgumentVariables& argument_variables,
      const GridlessBox& gridless_box,
      const DerivedCondition& derived_condition) noexcept {
    return derived_condition.dg_time_derivative(
        make_not_null(&get<EvolvedTags>(*mutated_variables))...,
        tuples::get<ArgumentTags>(argument_variables)...,
        db::get<GridlessTags>(gridless_box)...);
  }
};

template <typename GhCorrectionTempTagList,
          typename ValenciaCorrectionTempTagList, typename DerivedGhCondition,
          typename DerivedValenciaCondition, typename GridlessBox>
void test_boundary_condition_combination(
    const DerivedGhCondition& derived_gh_condition,
    const DerivedValenciaCondition& derived_valencia_condition,
    const grmhd::GhValenciaDivClean::BoundaryConditions::ProductOfConditions<
        DerivedGhCondition, DerivedValenciaCondition>&
        derived_product_condition,
    const GridlessBox& gridless_box) noexcept {
  using product_condition_type =
      grmhd::GhValenciaDivClean::BoundaryConditions::ProductOfConditions<
          DerivedGhCondition, DerivedValenciaCondition>;

  using gh_mutable_tags = tmpl::append<
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list,
      db::wrap_tags_in<
          ::Tags::Flux,
          typename GeneralizedHarmonic::System<3_st>::flux_variables,
          tmpl::size_t<3_st>, Frame::Inertial>,
      GhCorrectionTempTagList,
      tmpl::list<
          gr::Tags::InverseSpatialMetric<3_st, Frame::Inertial, DataVector>>>;
  using valencia_mutable_tags = tmpl::append<
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list,
      db::wrap_tags_in<::Tags::Flux,
                       typename grmhd::ValenciaDivClean::System::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      ValenciaCorrectionTempTagList,
      tmpl::list<
          gr::Tags::InverseSpatialMetric<3_st, Frame::Inertial, DataVector>>>;
  using product_mutable_tags = tmpl::append<
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list,
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list,
      db::wrap_tags_in<
          ::Tags::Flux,
          typename GeneralizedHarmonic::System<3_st>::flux_variables,
          tmpl::size_t<3_st>, Frame::Inertial>,
      db::wrap_tags_in<::Tags::Flux,
                       typename grmhd::ValenciaDivClean::System::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      tmpl::remove_duplicates<
          tmpl::append<GhCorrectionTempTagList, ValenciaCorrectionTempTagList>>,
      tmpl::list<
          gr::Tags::InverseSpatialMetric<3_st, Frame::Inertial, DataVector>>>;

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
  using valencia_arg_tags = tmpl::append<
      tmpl::list<::domain::Tags::MeshVelocity<3_st>,
                 evolution::dg::Tags::NormalCovector<3_st>,
                 evolution::dg::Actions::detail::NormalVector<3_st>>,
      typename DerivedValenciaCondition::dg_interior_evolved_variables_tags,
      typename DerivedValenciaCondition::dg_interior_primitive_variables_tags,
      typename DerivedValenciaCondition::dg_interior_primitive_variables_tags,
      typename DerivedValenciaCondition::dg_interior_temporary_tags,
      evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          DerivedValenciaCondition>,
      evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          DerivedValenciaCondition>>;
  using product_arg_tags = tmpl::append<
      tmpl::list<::domain::Tags::MeshVelocity<3_st>,
                 evolution::dg::Tags::NormalCovector<3_st>,
                 evolution::dg::Actions::detail::NormalVector<3_st>>,
      typename product_condition_type::dg_interior_evolved_variables_tags,
      typename product_condition_type::dg_interior_primitive_variables_tags,
      typename product_condition_type::dg_interior_primitive_variables_tags,
      typename product_condition_type::dg_interior_temporary_tags,
      typename product_condition_type::dg_interior_dt_vars_tags,
      typename product_condition_type::dg_interior_deriv_vars_tags>;

  using gh_gridless_tags = typename DerivedGhCondition::dg_gridless_tags;
  using valencia_gridless_tags =
      typename DerivedValenciaCondition::dg_gridless_tags;
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
                                    &argument_variables](auto tag_v) noexcept {
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

  using gh_bc_helper = ComputeBoundaryConditionHelper<
      DerivedGhCondition,
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list,
      gh_mutable_tags, gh_arg_tags, gh_gridless_tags>;
  using valencia_bc_helper = ComputeBoundaryConditionHelper<
      DerivedValenciaCondition,
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list,
      valencia_mutable_tags, valencia_arg_tags, valencia_gridless_tags>;
  using product_bc_helper = ComputeBoundaryConditionHelper<
      product_condition_type,
      tmpl::append<
          typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list,
          typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>,
      product_mutable_tags, product_arg_tags, product_gridless_tags>;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      product_condition_type::bc_type;

  if constexpr (bc_type == evolution::BoundaryConditions::Type::Ghost or
                bc_type == evolution::BoundaryConditions::Type::
                               GhostAndTimeDerivative) {
    auto gh_result = gh_bc_helper::dg_ghost(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_gh_condition);
    auto valencia_result = valencia_bc_helper::dg_ghost(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_valencia_condition);
    auto product_result = product_bc_helper::dg_ghost(
        make_not_null(&mutable_variables), argument_variables, gridless_box,
        derived_product_condition);
    if (gh_result.has_value()) {
      if (valencia_result.has_value()) {
        CHECK((product_result.value() ==
               gh_result.value() + ";" + valencia_result.value()));
      } else {
        CHECK(product_result == gh_result);
      }
    } else {
      CHECK(product_result == valencia_result);
    }
    CHECK_VARIABLES_APPROX(mutable_variables, expected_mutable_variables);
  }
  if constexpr (bc_type ==
                    evolution::BoundaryConditions::Type::TimeDerivative or
                bc_type == evolution::BoundaryConditions::Type::
                               GhostAndTimeDerivative) {
    auto gh_result = gh_bc_helper::dg_time_derivative(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_gh_condition);
    auto valencia_result = valencia_bc_helper::dg_time_derivative(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_valencia_condition);
    auto product_result = product_bc_helper::dg_time_derivative(
        make_not_null(&mutable_variables), argument_variables, gridless_box,
        derived_product_condition);
    if (gh_result.has_value()) {
      if (valencia_result.has_value()) {
        CHECK((product_result.value() ==
               gh_result.value() + ";" + valencia_result.value()));
      } else {
        CHECK(product_result == gh_result);
      }
    } else {
      CHECK(product_result == valencia_result);
    }
    CHECK_VARIABLES_APPROX(mutable_variables, expected_mutable_variables);
  }
  if constexpr (bc_type == evolution::BoundaryConditions::Type::Outflow) {
    auto gh_result = gh_bc_helper::dg_outflow(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_gh_condition);
    auto valencia_result = valencia_bc_helper::dg_outflow(
        make_not_null(&expected_mutable_variables), argument_variables,
        gridless_box, derived_valencia_condition);
    auto product_result = product_bc_helper::dg_outflow(
        make_not_null(&mutable_variables), argument_variables, gridless_box,
        derived_product_condition);
    if (gh_result.has_value()) {
      if (valencia_result.has_value()) {
        CHECK((product_result.value() ==
               gh_result.value() + ";" + valencia_result.value()));
      } else {
        CHECK(product_result == gh_result);
      }
    } else {
      CHECK(product_result == valencia_result);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GhValenciaDivClean.BoundaryConditions.ProductOfConditions",
    "[Unit][Evolution]") {
  // scoped to separate out each product combination
  {
    INFO("Product condition of DirichletAnalytic in each system");

    const GeneralizedHarmonic::BoundaryConditions::DirichletAnalytic<3_st>
        gh_condition{};
    const grmhd::ValenciaDivClean::BoundaryConditions::DirichletAnalytic
        valencia_condition{};
    const auto product_boundary_condition =
        TestHelpers::test_creation<std::unique_ptr<
            grmhd::GhValenciaDivClean::BoundaryConditions::BoundaryCondition>>(
            "ProductDirichletAnalyticAndDirichletAnalytic:\n"
            "  GeneralizedHarmonicDirichletAnalytic:\n"
            "  ValenciaDirichletAnalytic:");
    const auto gridless_box =
        db::create<db::AddSimpleTags<::Tags::Time, DummyAnalyticSolutionTag>>(
            0.5, GeneralizedHarmonic::Solutions::WrappedGr<
                     grmhd::Solutions::BondiMichel>{1.0, 4.0, 0.1, 2.0, 0.01});
    auto serialized_and_deserialized_condition = serialize_and_deserialize(
        *dynamic_cast<
            grmhd::GhValenciaDivClean::BoundaryConditions::ProductOfConditions<
                GeneralizedHarmonic::BoundaryConditions::DirichletAnalytic<
                    3_st>,
                grmhd::ValenciaDivClean::BoundaryConditions::
                    DirichletAnalytic>*>(product_boundary_condition.get()));
    test_boundary_condition_combination<
        GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<
            3_st>::dg_package_data_temporary_tags,
        grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov::
            dg_package_data_temporary_tags,
        GeneralizedHarmonic::BoundaryConditions::DirichletAnalytic<3_st>,
        grmhd::ValenciaDivClean::BoundaryConditions::DirichletAnalytic>(
        gh_condition, valencia_condition, serialized_and_deserialized_condition,
        gridless_box);
  }
}
