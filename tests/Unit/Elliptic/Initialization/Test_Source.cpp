// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Initialization/Source.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace {
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
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
struct Metavariables {
  using system = System<Dim>;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<AnalyticSolution<Dim>>;
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<analytic_solution_tag>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.Source",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    Mesh<1> mesh{3, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
    tnsr::I<DataVector, 1, Frame::Inertial> inertial_coords{{{{1., 2., 3.}}}};
    auto arguments_box =
        db::create<db::AddSimpleTags<Tags::Mesh<1>,
                                     Tags::Coordinates<1, Frame::Inertial>>>(
            mesh, std::move(inertial_coords));

    ActionTesting::MockRuntimeSystem<Metavariables<1>> runner{
        {AnalyticSolution<1>{}}, {}};

    const auto box =
        Elliptic::Initialization::Source<Metavariables<1>>::initialize(
            std::move(arguments_box), runner.cache());

    const DataVector source_expected{1., 2., 3.};
    CHECK(get<Tags::Source<ScalarFieldTag>>(box) ==
          Scalar<DataVector>(source_expected));
  }
  {
    INFO("2D");
    Mesh<2> mesh{{{3, 2}},
                 Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
    tnsr::I<DataVector, 2, Frame::Inertial> inertial_coords{
        {{{1., 2., 3., 4., 5., 6.}, {6, 0.}}}};
    auto arguments_box =
        db::create<db::AddSimpleTags<Tags::Mesh<2>,
                                     Tags::Coordinates<2, Frame::Inertial>>>(
            mesh, std::move(inertial_coords));

    ActionTesting::MockRuntimeSystem<Metavariables<2>> runner{
        {AnalyticSolution<2>{}}, {}};

    const auto box =
        Elliptic::Initialization::Source<Metavariables<2>>::initialize(
            std::move(arguments_box), runner.cache());

    const DataVector source_expected{1., 2., 3., 4., 5., 6.};
    CHECK(get<Tags::Source<ScalarFieldTag>>(box) ==
          Scalar<DataVector>(source_expected));
  }
  {
    INFO("3D");
    Mesh<3> mesh{{{3, 2, 2}},
                 Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
    tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{
        {{{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.},
          {12, 0.},
          {12, 0.}}}};
    auto arguments_box =
        db::create<db::AddSimpleTags<Tags::Mesh<3>,
                                     Tags::Coordinates<3, Frame::Inertial>>>(
            mesh, std::move(inertial_coords));

    ActionTesting::MockRuntimeSystem<Metavariables<3>> runner{
        {AnalyticSolution<3>{}}, {}};

    const auto box =
        Elliptic::Initialization::Source<Metavariables<3>>::initialize(
            std::move(arguments_box), runner.cache());

    const DataVector source_expected{1., 2., 3., 4.,  5.,  6.,
                                     7., 8., 9., 10., 11., 12.};
    CHECK(get<Tags::Source<ScalarFieldTag>>(box) ==
          Scalar<DataVector>(source_expected));
  }
}
