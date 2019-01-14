// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Initialization/LinearSolver.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel

namespace {
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

struct LinearSolverSourceTag : db::SimpleTag {
  static std::string name() noexcept { return "LinearSolverSourceTag"; };
  using type = Scalar<DataVector>;
};

struct LinearSolverAxTag : db::SimpleTag {
  static std::string name() noexcept { return "LinearSolverAxTag"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;
  struct linear_solver {
    struct tags {
      using simple_tags =
          db::AddSimpleTags<LinearSolverSourceTag, LinearSolverAxTag>;
      using compute_tags = db::AddComputeTags<>;
      template <typename TagsList, typename ArrayIndex,
                typename ParallelComponent>
      static auto initialize(
          db::DataBox<TagsList>&& box,
          const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
          const ArrayIndex& /*array_index*/,
          const ParallelComponent* const /*meta*/,
          const db::item_type<
              db::add_tag_prefix<::Tags::Source, typename system::fields_tag>>&
              b,
          const db::item_type<
              db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                                 typename system::fields_tag>>& Ax) noexcept {
        return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
            std::move(box), get<::Tags::Source<ScalarFieldTag>>(b),
            get<LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>>(Ax));
      }
    };
  };
};

struct MockParallelComponent {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.LinearSolver",
                  "[Unit][Elliptic][Actions]") {
  SECTION("1D") {
    db::item_type<
        db::add_tag_prefix<Tags::Source, typename System<1>::fields_tag>>
        sources{3, 0.};
    get<Tags::Source<ScalarFieldTag>>(sources) =
        Scalar<DataVector>{{{{1., 2., 3.}}}};
    auto arguments_box = db::create<db::AddSimpleTags<
        db::add_tag_prefix<Tags::Source, typename System<1>::fields_tag>>>(
        std::move(sources));

    ActionTesting::MockRuntimeSystem<Metavariables<1>> runner{{}, {}};
    MockParallelComponent component{};
    const auto box =
        Elliptic::Initialization::LinearSolver<Metavariables<1>>::initialize(
            std::move(arguments_box), runner.cache(), 0, &component);

    const DataVector b_expected{1., 2., 3.};
    CHECK(get<LinearSolverSourceTag>(box) == Scalar<DataVector>(b_expected));
    const DataVector Ax_expected(3, 0.);
    CHECK(get<LinearSolverAxTag>(box) == Scalar<DataVector>(Ax_expected));
  }
//   SECTION("2D") {
//     Mesh<2> mesh{{{3, 2}},
//                  Spectral::Basis::Legendre,
//                  Spectral::Quadrature::GaussLobatto};
//     tnsr::I<DataVector, 2, Frame::Inertial> inertial_coords{
//         {{{1., 2., 3., 4., 5., 6.}, {6, 0.}}}};
//     auto arguments_box =
//         db::create<db::AddSimpleTags<Tags::Mesh<2>,
//                                      Tags::Coordinates<2, Frame::Inertial>>>(
//             mesh, std::move(inertial_coords));

//     ActionTesting::MockRuntimeSystem<Metavariables<2>> runner{
//         {AnalyticSolution<2>{}}, {}};

//     MockParallelComponent component{};
//     const auto box =
//         Elliptic::Initialization::LinearSolver<Metavariables<2>>::initialize(
//             std::move(arguments_box), runner.cache(), 0, &component);

//     const DataVector b_expected{1., 2., 3., 4., 5., 6.};
//     CHECK(get<LinearSolverSourceTag>(box) == Scalar<DataVector>(b_expected));
//     const DataVector Ax_expected(6, 0.);
//     CHECK(get<LinearSolverAxTag>(box) == Scalar<DataVector>(Ax_expected));
//   }
//   SECTION("3D") {
//     Mesh<3> mesh{{{3, 2, 2}},
//                  Spectral::Basis::Legendre,
//                  Spectral::Quadrature::GaussLobatto};
//     tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{
//         {{{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.},
//           {12, 0.},
//           {12, 0.}}}};
//     auto arguments_box =
//         db::create<db::AddSimpleTags<Tags::Mesh<3>,
//                                      Tags::Coordinates<3, Frame::Inertial>>>(
//             mesh, std::move(inertial_coords));

//     ActionTesting::MockRuntimeSystem<Metavariables<3>> runner{
//         {AnalyticSolution<3>{}}, {}};

//     MockParallelComponent component{};
//     const auto box =
//         Elliptic::Initialization::LinearSolver<Metavariables<3>>::initialize(
//             std::move(arguments_box), runner.cache(), 0, &component);

//     const DataVector b_expected{1., 2., 3., 4.,  5.,  6.,
//                                 7., 8., 9., 10., 11., 12.};
//     CHECK(get<LinearSolverSourceTag>(box) == Scalar<DataVector>(b_expected));
//     const DataVector Ax_expected(12, 0.);
//     CHECK(get<LinearSolverAxTag>(box) == Scalar<DataVector>(Ax_expected));
//   }
}
