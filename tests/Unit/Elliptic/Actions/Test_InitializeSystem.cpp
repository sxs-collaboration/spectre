// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Actions/InitializeDomain.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/ElementId.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <typename Tag>
struct SomePrefix : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "SomePrefix"; }
  using type = db::item_type<Tag>;
  using tag = Tag;
};

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<Tags::Source<ScalarFieldTag>> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<Tags::Source<ScalarFieldTag>> /*meta*/) const noexcept {
    return {Scalar<DataVector>(get<0>(x))};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  using variables_tag = db::add_tag_prefix<SomePrefix, fields_tag>;
  using gradient_tags = tmpl::list<ScalarFieldTag>;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<::Tags::Domain<Dim, Frame::Inertial>,
                                    ::Tags::InitialExtents<Dim>>>,
                     domain::Actions::InitializeDomain<Dim>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<elliptic::Actions::InitializeSystem>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<AnalyticSolution<Dim>>;
  using const_global_cache_tag_list = tmpl::list<analytic_solution_tag>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeSystem",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{1.}}, {{3.}}, {{false}}, {{0}}, {{3}}};
    const ElementId<1> element_id{0};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<1>{}}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents()});
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

    const Scalar<DataVector> expected_initial_field{{{{3, 0.}}}};
    CHECK(get_tag(ScalarFieldTag{}) == expected_initial_field);
    // Only check that the variables are initialized
    get_tag(SomePrefix<ScalarFieldTag>{});
    get_tag(
        LinearSolver::Tags::OperatorAppliedTo<SomePrefix<ScalarFieldTag>>{});

    const DataVector source_expected{1., 2., 3.};
    CHECK(get_tag(Tags::Source<ScalarFieldTag>{}) ==
          Scalar<DataVector>(source_expected));
  }
  {
    INFO("2D");
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{1., 0.}}, {{3., 1.}}, {{false, false}}, {{0, 0}}, {{3, 2}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::ProductOf2Maps<
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>>));

    const ElementId<2> element_id{0};
    using metavariables = Metavariables<2>;
    using element_array = ElementArray<2, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<2>{}}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents()});
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

    const Scalar<DataVector> expected_initial_field{{{{6, 0.}}}};
    CHECK(get_tag(ScalarFieldTag{}) == expected_initial_field);
    // Only check that the variables are initialized
    get_tag(SomePrefix<ScalarFieldTag>{});
    get_tag(
        LinearSolver::Tags::OperatorAppliedTo<SomePrefix<ScalarFieldTag>>{});

    const DataVector source_expected{1., 2., 3., 1., 2., 3.};
    CHECK(get_tag(Tags::Source<ScalarFieldTag>{}) ==
          Scalar<DataVector>(source_expected));
  }
  {
    INFO("3D");
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{1., 0., 0.}},
        {{3., 1., 1.}},
        {{false, false, false}},
        {{0, 0, 0}},
        {{3, 2, 2}}};
    // Register the coordinate map for serialization
    PUPable_reg(SINGLE_ARG(
        domain::CoordinateMap<
            Frame::Logical, Frame::Inertial,
            domain::CoordinateMaps::ProductOf3Maps<
                domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
                domain::CoordinateMaps::Affine>>));

    const ElementId<3> element_id{0};
    using metavariables = Metavariables<3>;
    using element_array = ElementArray<3, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<3>{}}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents()});
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

    const Scalar<DataVector> expected_initial_field{{{{12, 0.}}}};
    CHECK(get_tag(ScalarFieldTag{}) == expected_initial_field);
    // Only check that the variables are initialized
    get_tag(SomePrefix<ScalarFieldTag>{});
    get_tag(
        LinearSolver::Tags::OperatorAppliedTo<SomePrefix<ScalarFieldTag>>{});

    const DataVector source_expected{1., 2., 3., 1., 2., 3.,
                                     1., 2., 3., 1., 2., 3.};
    CHECK(get_tag(Tags::Source<ScalarFieldTag>{}) ==
          Scalar<DataVector>(source_expected));
  }
}
