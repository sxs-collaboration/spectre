// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/ProjectSpectralFilters.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using TagList = tmpl::list<ScalarVar>;
constexpr size_t Dim = 1;
using FilterTag = Filters::Tags::SpectralFilter<Dim, TagList>;
using Projector =
    evolution::dg::Initialization::ProjectSpectralFilters<Dim, TagList>;

std::unique_ptr<Filters::Filter<Dim, TagList>> make_none() {
  return std::make_unique<Filters::None<Dim, TagList>>();
}

std::unique_ptr<Filters::Filter<Dim, TagList>> make_hypercube() {
  return std::make_unique<Filters::Hypercube<Dim, TagList>>(
      4u, true, std::nullopt, false, false, std::nullopt, std::nullopt);
}

void test_p_refinement() {
  const ElementId<Dim> element_id{0};
  const Element<Dim> element{element_id, DirectionMap<Dim, Neighbors<Dim>>{}};
  const Mesh<Dim> mesh{2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  auto box = db::create<db::AddSimpleTags<FilterTag>>(make_none());
  const Filters::Filter<Dim, TagList>* const ptr_before =
      &db::get<FilterTag>(box);
  db::mutate_apply<Projector>(make_not_null(&box),
                              std::make_pair(mesh, element));
  CHECK(&db::get<FilterTag>(box) == ptr_before);
  CHECK(db::get<FilterTag>(box).is_equal(Filters::None<Dim, TagList>{}));
}

void test_splitting() {
  auto box = db::DataBox<tmpl::list<FilterTag>>{};
  tuples::TaggedTuple<FilterTag> parent_items{make_none()};
  db::mutate_apply<Projector>(make_not_null(&box), parent_items);
  CHECK(db::get<FilterTag>(box).is_equal(Filters::None<Dim, TagList>{}));
}

void test_joining() {
  auto box = db::DataBox<tmpl::list<FilterTag>>{};
  std::unordered_map<ElementId<Dim>, tuples::TaggedTuple<FilterTag>>
      children_items;
  children_items.emplace(ElementId<Dim>{0},
                         tuples::TaggedTuple<FilterTag>{make_none()});
  children_items.emplace(ElementId<Dim>{1},
                         tuples::TaggedTuple<FilterTag>{make_none()});
  db::mutate_apply<Projector>(make_not_null(&box), children_items);
  CHECK(db::get<FilterTag>(box).is_equal(Filters::None<Dim, TagList>{}));
}

void test_joining_error() {
  CHECK_THROWS_WITH(
      ([]() {
        auto box = db::DataBox<tmpl::list<FilterTag>>{};
        std::unordered_map<ElementId<Dim>, tuples::TaggedTuple<FilterTag>>
            children_items;
        children_items.emplace(ElementId<Dim>{0},
                               tuples::TaggedTuple<FilterTag>{make_none()});
        children_items.emplace(
            ElementId<Dim>{1},
            tuples::TaggedTuple<FilterTag>{make_hypercube()});
        db::mutate_apply<Projector>(make_not_null(&box), children_items);
      })(),
      Catch::Matchers::ContainsSubstring("Children do not agree"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Initialization.ProjectSpectralFilters",
                  "[Unit][Evolution]") {
  static_assert(tt::assert_conforms_to_v<Projector, amr::protocols::Projector>);
  test_p_refinement();
  test_splitting();
  test_joining();
  test_joining_error();
}
