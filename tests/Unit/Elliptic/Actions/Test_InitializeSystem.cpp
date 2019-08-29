// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  using primal_fields = tmpl::list<ScalarFieldTag>;
  using gradient_tags = tmpl::list<LinearSolver::Tags::Operand<ScalarFieldTag>>;
};

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<Tags::Source<ScalarFieldTag>> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<Tags::Source<ScalarFieldTag>> /*meta*/) const noexcept {
    Scalar<DataVector> source{get<0>(x)};
    for (size_t d = 1; d < Dim; d++) {
      get(source) += x.get(d);
    }
    return {source};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tags =
      tmpl::list<::Tags::Domain<Dim, Frame::Inertial>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<::Tags::InitialExtents<Dim>>>,
                     dg::Actions::InitializeDomain<Dim>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<elliptic::Actions::InitializeSystem>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using analytic_solution_tag = Tags::AnalyticSolution<AnalyticSolution<Dim>>;
  using const_global_cache_tags = tmpl::list<analytic_solution_tag>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim>
void check_compute_items(
    const ActionTesting::MockRuntimeSystem<Metavariables<Dim>>& runner,
    const ElementId<Dim>& element_id) {
  // The compute items themselves are tested elsewhere, so just check if they
  // were indeed added by the initializer
  const auto tag_is_retrievable = [&runner,
                                   &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::tag_is_retrievable<
        ElementArray<Dim, Metavariables<Dim>>, tag>(runner, element_id);
  };
  CHECK(tag_is_retrievable(
      ::Tags::deriv<LinearSolver::Tags::Operand<ScalarFieldTag>,
                    tmpl::size_t<Dim>, Frame::Inertial>{}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeSystem",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Which element we work with does not matter for this test
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<1>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    check_compute_items(runner, element_id);

    // Test the initial guess is zero
    CHECK(get_tag(ScalarFieldTag{}) == Scalar<DataVector>{{{{4, 0.}}}});
    CHECK(get_tag(LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>{}) ==
          Scalar<DataVector>{{{{4, 0.}}}});
    // Test the analytic source
    const auto& inertial_coords =
        get_tag(Tags::Coordinates<1, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::Source<ScalarFieldTag>{})) ==
          // This check is against the source computed by the
          // analytic solution above
          get<0>(inertial_coords));
    // Test the linear solver quantities are initialized, but value is undefined
    CHECK(get(get_tag(LinearSolver::Tags::Operand<ScalarFieldTag>{})).size() ==
          4);
    CHECK(get(get_tag(LinearSolver::Tags::OperatorAppliedTo<
                      LinearSolver::Tags::Operand<ScalarFieldTag>>{}))
              .size() == 4);
  }
  {
    INFO("2D");
    // Which element we work with does not matter for this test
    const ElementId<2> element_id{0, {{SegmentId{2, 1}, SegmentId{0, 0}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 2.}}, {{false, false}}, {{2, 0}}, {{4, 3}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::ProductOf2Maps<
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<2>;
    using element_array = ElementArray<2, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<2>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    check_compute_items(runner, element_id);

    // Test the initial guess is zero
    CHECK(get_tag(ScalarFieldTag{}) == Scalar<DataVector>{{{{12, 0.}}}});
    CHECK(get_tag(LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>{}) ==
          Scalar<DataVector>{{{{12, 0.}}}});
    // Test the analytic source
    const auto& inertial_coords =
        get_tag(Tags::Coordinates<2, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::Source<ScalarFieldTag>{})) ==
          // This check is against the source computed by the
          // analytic solution above
          get<0>(inertial_coords) + get<1>(inertial_coords));
    CHECK(get(get_tag(LinearSolver::Tags::Operand<ScalarFieldTag>{})).size() ==
          12);
    CHECK(get(get_tag(LinearSolver::Tags::OperatorAppliedTo<
                      LinearSolver::Tags::Operand<ScalarFieldTag>>{}))
              .size() == 12);
  }
  {
    INFO("3D");
    // Which element we work with does not matter for this test
    const ElementId<3> element_id{
        0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 1}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 2., 3.}},
        {{false, false, false}},
        {{2, 0, 1}},
        {{4, 3, 2}}};
    // Register the coordinate map for serialization
    PUPable_reg(SINGLE_ARG(
        domain::CoordinateMap<
            Frame::Logical, Frame::Inertial,
            domain::CoordinateMaps::ProductOf3Maps<
                domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
                domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<3>;
    using element_array = ElementArray<3, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<3>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    check_compute_items(runner, element_id);

    // Test the initial guess is zero
    CHECK(get_tag(ScalarFieldTag{}) == Scalar<DataVector>{{{{24, 0.}}}});
    CHECK(get_tag(LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>{}) ==
          Scalar<DataVector>{{{{24, 0.}}}});
    // Test the analytic source
    const auto& inertial_coords =
        get_tag(Tags::Coordinates<3, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::Source<ScalarFieldTag>{})) ==
          // This check is against the source computed by the
          // analytic solution above
          get<0>(inertial_coords) + get<1>(inertial_coords) +
              get<2>(inertial_coords));
    // Test the linear solver quantities are initialized, but value is undefined
    CHECK(get(get_tag(LinearSolver::Tags::Operand<ScalarFieldTag>{})).size() ==
          24);
    CHECK(get(get_tag(LinearSolver::Tags::OperatorAppliedTo<
                      LinearSolver::Tags::Operand<ScalarFieldTag>>{}))
              .size() == 24);
  }
}
