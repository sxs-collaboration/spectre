// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct DummyOptionsGroup {};

template <size_t N>
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using fields_tag =
    ::Tags::Variables<tmpl::list<ScalarFieldTag<0>, ScalarFieldTag<1>>>;

template <size_t Dim, bool RestrictToOverlap, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
              domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
              LinearSolver::Schwarz::Tags::IntrudingExtents<Dim,
                                                            DummyOptionsGroup>,
              Convergence::Tags::IterationId<DummyOptionsGroup>, fields_tag,
              LinearSolver::Schwarz::Tags::Overlaps<fields_tag, Dim,
                                                    DummyOptionsGroup>>>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              LinearSolver::Schwarz::Actions::SendOverlapFields<
                  tmpl::list<fields_tag>, DummyOptionsGroup, RestrictToOverlap>,
              LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
                  Dim, tmpl::list<fields_tag>, DummyOptionsGroup>,
              Parallel::Actions::TerminatePhase>>>;
};

template <size_t Dim, bool RestrictToOverlap>
struct Metavariables {
  using element_array = ElementArray<Dim, RestrictToOverlap, Metavariables>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim, bool RestrictToOverlap>
void test_communicate_overlap_fields(const size_t num_points_per_dim,
                                     const size_t overlap,
                                     const DataVector& overlap_data,
                                     const DataVector& expected_overlap_data) {
  CAPTURE(Dim);
  CAPTURE(RestrictToOverlap);
  CAPTURE(overlap);

  using metavariables = Metavariables<Dim, RestrictToOverlap>;
  using element_array = typename metavariables::element_array;

  ActionTesting::MockRuntimeSystem<metavariables> runner{tuples::TaggedTuple<
      LinearSolver::Schwarz::Tags::MaxOverlap<DummyOptionsGroup>,
      logging::Tags::Verbosity<DummyOptionsGroup>>{overlap,
                                                   Verbosity::Verbose}};

  // Setup element array
  const auto add_element = [num_points_per_dim, overlap, &overlap_data,
                            &runner](const Element<Dim>& element) {
    Mesh<Dim> mesh{num_points_per_dim, Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    auto intruding_extents = make_array<Dim>(overlap);
    typename fields_tag::type fields{mesh.number_of_grid_points()};
    get(get<ScalarFieldTag<0>>(fields)) = overlap_data;
    get(get<ScalarFieldTag<1>>(fields)) = overlap_data;
    ActionTesting::emplace_component_and_initialize<element_array>(
        make_not_null(&runner), element.id(),
        {element, std::move(mesh), std::move(intruding_extents), size_t{0},
         std::move(fields),
         LinearSolver::Schwarz::OverlapMap<Dim, typename fields_tag::type>{}});
  };
  const ElementId<Dim> first_element_id{0};
  const ElementId<Dim> second_element_id{1};
  const ElementId<Dim> third_element_id{2};
  // We can only have multiple face-neighbors in >1 dimensions
  if constexpr (Dim > 1) {
    add_element({first_element_id,
                 {{Direction<Dim>::upper_xi(),
                   {{second_element_id, third_element_id}, {}}}}});
    add_element({second_element_id,
                 {{Direction<Dim>::lower_xi(), {{first_element_id}, {}}}}});
    add_element({third_element_id,
                 {{Direction<Dim>::lower_xi(), {{first_element_id}, {}}}}});
  } else {
    add_element({first_element_id,
                 {{Direction<Dim>::upper_xi(), {{second_element_id}, {}}}}});
    add_element({second_element_id,
                 {{Direction<Dim>::lower_xi(), {{first_element_id}, {}}}}});
  }

  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  // Send from first element
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            first_element_id);
  REQUIRE_FALSE(
      ActionTesting::is_ready<element_array>(runner, first_element_id));
  // Send from second element
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            second_element_id);
  REQUIRE(ActionTesting::is_ready<element_array>(runner, first_element_id) ==
          (Dim == 1));
  REQUIRE(ActionTesting::is_ready<element_array>(runner, second_element_id));
  // Send from third element
  if constexpr (Dim > 1) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              third_element_id);
    REQUIRE(ActionTesting::is_ready<element_array>(runner, first_element_id));
    REQUIRE(ActionTesting::is_ready<element_array>(runner, third_element_id));
  }
  // Receive on first element
  {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              first_element_id);
    const auto& all_overlap_fields =
        ActionTesting::get_databox_tag<element_array,
                                       LinearSolver::Schwarz::Tags::Overlaps<
                                           fields_tag, Dim, DummyOptionsGroup>>(
            runner, first_element_id);
    CHECK(all_overlap_fields.size() == (Dim > 1 ? 2 : 1));
    const auto& overlap_fields_on_second_element =
        all_overlap_fields.at({Direction<Dim>::upper_xi(), second_element_id});
    CHECK_ITERABLE_APPROX(
        get(get<ScalarFieldTag<0>>(overlap_fields_on_second_element)),
        expected_overlap_data);
    CHECK_ITERABLE_APPROX(
        get(get<ScalarFieldTag<1>>(overlap_fields_on_second_element)),
        expected_overlap_data);
    if constexpr (Dim > 1) {
      const auto& overlap_fields_on_third_element =
          all_overlap_fields.at({Direction<Dim>::upper_xi(), third_element_id});
      CHECK_ITERABLE_APPROX(
          get(get<ScalarFieldTag<0>>(overlap_fields_on_third_element)),
          expected_overlap_data);
      CHECK_ITERABLE_APPROX(
          get(get<ScalarFieldTag<1>>(overlap_fields_on_third_element)),
          expected_overlap_data);
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.Action.CommunicateOverlapFields",
                  "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  test_communicate_overlap_fields<1, false>(3, 2, {0., 1., 2.}, {0., 1., 2.});
  test_communicate_overlap_fields<1, true>(3, 2, {0., 1., 2.}, {0., 1.});
  test_communicate_overlap_fields<2, false>(
      3, 2, {0., 1., 2., 3., 4., 5., 6., 7., 8.},
      {0., 1., 2., 3., 4., 5., 6., 7., 8.});
  test_communicate_overlap_fields<2, true>(
      3, 2, {0., 1., 2., 3., 4., 5., 6., 7., 8.}, {0., 1., 3., 4., 6., 7.});
  test_communicate_overlap_fields<3, false>(
      2, 1, {0., 1., 2., 3., 4., 5., 6., 7.}, {0., 1., 2., 3., 4., 5., 6., 7.});
  test_communicate_overlap_fields<3, true>(
      2, 1, {0., 1., 2., 3., 4., 5., 6., 7.}, {0., 2., 4., 6.});
}
