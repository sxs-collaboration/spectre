// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct TagInt : db::SimpleTag {
  using type = int;
};

struct TagDouble : db::SimpleTag {
  using type = double;
};

void test_p_refinement() {
  const ElementId<1> element_id{0};
  Element<1> element{element_id, DirectionMap<1, Neighbors<1>>{}};
  const Mesh<1> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  auto box = db::create<db::AddSimpleTags<TagInt, TagDouble>>(3, 4.2);
  db::mutate_apply<
      amr::projectors::CopyFromCreatorOrLeaveAsIs<TagInt, TagDouble>>(
      make_not_null(&box), std::make_pair(mesh, element));
  CHECK(db::get<TagInt>(box) == 3);
  CHECK(db::get<TagDouble>(box) == 4.2);
}

void test_splitting() {
  auto box = db::DataBox<tmpl::list<TagInt, TagDouble>>{};
  tuples::TaggedTuple<TagInt, TagDouble> parent_items{3, 4.2};
  db::mutate_apply<amr::projectors::CopyFromCreatorOrLeaveAsIs<
      tmpl::list<TagInt, TagDouble>>>(make_not_null(&box),
                                                    parent_items);
  CHECK(db::get<TagInt>(box) == 3);
  CHECK(db::get<TagDouble>(box) == 4.2);
}

void test_joining() {
  auto box = db::DataBox<tmpl::list<TagInt, TagDouble>>{};
  tuples::TaggedTuple<TagInt, TagDouble> child_one_items{3, 4.2};
  tuples::TaggedTuple<TagInt, TagDouble> child_two_items{3, 4.2};
  const std::unordered_map<ElementId<1>, tuples::TaggedTuple<TagInt, TagDouble>>
      children_items{{ElementId<1>{0}, child_one_items},
                     {ElementId<1>{1}, child_two_items}};
  db::mutate_apply<amr::projectors::CopyFromCreatorOrLeaveAsIs<
      tmpl::list<TagInt, TagDouble>>>(make_not_null(&box), children_items);
  CHECK(db::get<TagInt>(box) == 3);
  CHECK(db::get<TagDouble>(box) == 4.2);
}

void test_joining_assert() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        auto box = db::DataBox<tmpl::list<TagInt, TagDouble>>{};
        tuples::TaggedTuple<TagInt, TagDouble> child_one_items{3, 4.2};
        tuples::TaggedTuple<TagInt, TagDouble> child_two_items{3, 4.1};
        const std::unordered_map<ElementId<1>,
                                 tuples::TaggedTuple<TagInt, TagDouble>>
            children_items{{ElementId<1>{0}, child_one_items},
                           {ElementId<1>{1}, child_two_items}};
        db::mutate_apply<amr::projectors::CopyFromCreatorOrLeaveAsIs<
            tmpl::list<TagInt, TagDouble>>>(make_not_null(&box),
                                            children_items);
      })(),
      Catch::Matchers::ContainsSubstring("Children do not agree"));
#endif  // #ifdef SPECTRE_DEBUG
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Projectors.CopyFromCreatorOrLeaveAsIs",
                  "[Domain][Unit]") {
  static_assert(
      tt::assert_conforms_to_v<amr::projectors::CopyFromCreatorOrLeaveAsIs<
                                   tmpl::list<TagInt, TagDouble>>,
                               amr::protocols::Projector>);
  test_p_refinement();
  test_splitting();
  test_joining();
  test_joining_assert();
}
