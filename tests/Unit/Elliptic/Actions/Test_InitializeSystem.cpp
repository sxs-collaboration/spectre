// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AuxiliaryFieldTag : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag =
      Tags::Variables<tmpl::list<ScalarFieldTag, AuxiliaryFieldTag<Dim>>>;
  using primal_fields = tmpl::list<ScalarFieldTag>;
  using auxiliary_fields = tmpl::list<AuxiliaryFieldTag<Dim>>;
};

template <size_t Dim>
using linear_operator_applied_to_fields_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                       typename System<Dim>::fields_tag>;

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<Tags::FixedSource<ScalarFieldTag>> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<Tags::FixedSource<ScalarFieldTag>> /*meta*/) const noexcept {
    Scalar<DataVector> fixed_source{get<0>(x)};
    for (size_t d = 1; d < Dim; d++) {
      get(fixed_source) += x.get(d);
    }
    return {fixed_source};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<
                  tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                             domain::Tags::InitialExtents<Dim>,
                             linear_operator_applied_to_fields_tag<Dim>>>,
              Actions::SetupDataBox, dg::Actions::InitializeDomain<Dim>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::SetupDataBox,
                                        elliptic::Actions::InitializeSystem<
                                            typename Metavariables::system>>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using analytic_solution_tag = Tags::AnalyticSolution<AnalyticSolution<Dim>>;
  using const_global_cache_tags = tmpl::list<analytic_solution_tag>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeSystem",
                  "[Unit][Elliptic][Actions]") {
  domain::creators::register_derived_with_charm();
  {
    INFO("1D");
    // Which element we work with does not matter for this test
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{2}}, {{4}}, {{false}}, nullptr};

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<1>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.initial_refinement_levels(),
         domain_creator.initial_extents(),
         typename linear_operator_applied_to_fields_tag<1>::type{4}});
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // Test the initial guess is zero
    CHECK(get_tag(ScalarFieldTag{}) == Scalar<DataVector>{{{{4, 0.}}}});
    CHECK(get_tag(LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>{}) ==
          Scalar<DataVector>{{{{4, 0.}}}});
    // Test the analytic source
    const auto& inertial_coords =
        get_tag(domain::Tags::Coordinates<1, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::FixedSource<ScalarFieldTag>{})) ==
          // This check is against the source computed by the
          // analytic solution above
          get<0>(inertial_coords));
  }
  {
    INFO("2D");
    // Which element we work with does not matter for this test
    const ElementId<2> element_id{0, {{SegmentId{2, 1}, SegmentId{0, 0}}}};
    const domain::creators::Rectangle domain_creator{
        {{-0.5, 0.}}, {{1.5, 2.}}, {{2, 0}}, {{4, 3}}, {{false, false}}};

    using metavariables = Metavariables<2>;
    using element_array = ElementArray<2, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<2>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.initial_refinement_levels(),
         domain_creator.initial_extents(),
         typename linear_operator_applied_to_fields_tag<2>::type{12}});
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // Test the initial guess is zero
    CHECK(get_tag(ScalarFieldTag{}) == Scalar<DataVector>{{{{12, 0.}}}});
    CHECK(get_tag(LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>{}) ==
          Scalar<DataVector>{{{{12, 0.}}}});
    // Test the analytic source
    const auto& inertial_coords =
        get_tag(domain::Tags::Coordinates<2, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::FixedSource<ScalarFieldTag>{})) ==
          // This check is against the source computed by the
          // analytic solution above
          get<0>(inertial_coords) + get<1>(inertial_coords));
  }
  {
    INFO("3D");
    // Which element we work with does not matter for this test
    const ElementId<3> element_id{
        0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 1}}}};
    const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
                                                 {{1.5, 2., 3.}},
                                                 {{false, false, false}},
                                                 {{2, 0, 1}},
                                                 {{4, 3, 2}}};

    using metavariables = Metavariables<3>;
    using element_array = ElementArray<3, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<3>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.initial_refinement_levels(),
         domain_creator.initial_extents(),
         typename linear_operator_applied_to_fields_tag<3>::type{24}});
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // Test the initial guess is zero
    CHECK(get_tag(ScalarFieldTag{}) == Scalar<DataVector>{{{{24, 0.}}}});
    CHECK(get_tag(LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>{}) ==
          Scalar<DataVector>{{{{24, 0.}}}});
    // Test the analytic source
    const auto& inertial_coords =
        get_tag(domain::Tags::Coordinates<3, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::FixedSource<ScalarFieldTag>{})) ==
          // This check is against the source computed by the
          // analytic solution above
          get<0>(inertial_coords) + get<1>(inertial_coords) +
              get<2>(inertial_coords));
  }
}
