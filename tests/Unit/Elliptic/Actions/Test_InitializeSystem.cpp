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
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
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
struct Fluxes {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const tnsr::i<DataVector, Dim>& auxiliary_field) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_field->get(d) = auxiliary_field.get(d);
    }
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_aux_field,
      const Scalar<DataVector>& field) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_aux_field->get(d, d) = get(field);
    }
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct Sources {
  using argument_tags = tmpl::list<>;
  static void apply(const gsl::not_null<Scalar<DataVector>*> source_for_field,
                    const Scalar<DataVector>& field) {
    get(*source_for_field) = get(field);
  }
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag =
      Tags::Variables<tmpl::list<ScalarFieldTag, AuxiliaryFieldTag<Dim>>>;
  using primal_fields = tmpl::list<ScalarFieldTag>;
  using auxiliary_fields = tmpl::list<AuxiliaryFieldTag<Dim>>;
  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using primal_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, primal_fields>;
  using auxiliary_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, auxiliary_fields>;
  using fluxes = Fluxes<Dim>;
  using sources = Sources;
};

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
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                                    domain::Tags::InitialExtents<Dim>>>,
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
  using const_global_cache_tags =
      tmpl::list<analytic_solution_tag,
                 elliptic::Tags::FluxesComputer<Fluxes<Dim>>>;
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
      ::Tags::Flux<LinearSolver::Tags::Operand<ScalarFieldTag>,
                   tmpl::size_t<Dim>, Frame::Inertial>{}));
  CHECK(tag_is_retrievable(
      ::Tags::Flux<LinearSolver::Tags::Operand<AuxiliaryFieldTag<Dim>>,
                   tmpl::size_t<Dim>, Frame::Inertial>{}));
  CHECK(tag_is_retrievable(
      ::Tags::div<::Tags::Flux<LinearSolver::Tags::Operand<ScalarFieldTag>,
                               tmpl::size_t<Dim>, Frame::Inertial>>{}));
  CHECK(tag_is_retrievable(
      ::Tags::div<
          ::Tags::Flux<LinearSolver::Tags::Operand<AuxiliaryFieldTag<Dim>>,
                       tmpl::size_t<Dim>, Frame::Inertial>>{}));
  CHECK(tag_is_retrievable(
      ::Tags::Source<LinearSolver::Tags::Operand<ScalarFieldTag>>{}));
  CHECK(tag_is_retrievable(
      ::Tags::Source<LinearSolver::Tags::Operand<AuxiliaryFieldTag<Dim>>>{}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeSystem",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Which element we work with does not matter for this test
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<1>{}, Fluxes<1>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_refinement_levels(),
                              domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
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
        get_tag(domain::Tags::Coordinates<1, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::FixedSource<ScalarFieldTag>{})) ==
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
    const domain::creators::Rectangle domain_creator{
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
        {AnalyticSolution<2>{}, Fluxes<2>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_refinement_levels(),
                              domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
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
        get_tag(domain::Tags::Coordinates<2, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::FixedSource<ScalarFieldTag>{})) ==
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
    const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
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
        {AnalyticSolution<3>{}, Fluxes<3>{}, domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_refinement_levels(),
                              domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
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
        get_tag(domain::Tags::Coordinates<3, Frame::Inertial>{});
    CHECK(get(get_tag(Tags::FixedSource<ScalarFieldTag>{})) ==
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
