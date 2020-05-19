// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<ScalarFieldTag> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<ScalarFieldTag> /*meta*/) const noexcept {
    Scalar<DataVector> solution{2. * get<0>(x)};
    for (size_t d = 1; d < Dim; d++) {
      get(solution) += 2. * x.get(d);
    }
    return {std::move(solution)};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct AnalyticSolutionTag : db::SimpleTag {
  using type = AnalyticSolution<Dim>;
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::Actions::InitializeAnalyticSolution<
              AnalyticSolutionTag<Dim>, tmpl::list<ScalarFieldTag>>>>>;
};

template <size_t Dim>
struct Metavariables {
  using element_array = ElementArray<Dim, Metavariables>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim>
void test_initialize_analytic_solution(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const Scalar<DataVector>& expected_solution) {
  using metavariables = Metavariables<Dim>;
  using element_array = typename metavariables::element_array;
  const ElementId<Dim> element_id{0};
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {AnalyticSolution<Dim>{}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id, {inertial_coords});
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  CHECK_ITERABLE_APPROX(
      get(ActionTesting::get_databox_tag<element_array,
                                         ::Tags::Analytic<ScalarFieldTag>>(
          runner, element_id)),
      get(expected_solution));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeAnalyticSolution",
                  "[Unit][Elliptic][Actions]") {
  test_initialize_analytic_solution(
      tnsr::I<DataVector, 1, Frame::Inertial>{{{{1., 2., 3., 4.}}}},
      Scalar<DataVector>{{{{2., 4., 6., 8.}}}});
  test_initialize_analytic_solution(
      tnsr::I<DataVector, 2, Frame::Inertial>{{{{1., 2.}, {3., 4.}}}},
      Scalar<DataVector>{{{{8., 12.}}}});
  test_initialize_analytic_solution(
      tnsr::I<DataVector, 3, Frame::Inertial>{{{{1., 2.}, {3., 4.}, {5., 6.}}}},
      Scalar<DataVector>{{{{18., 24.}}}});
}
