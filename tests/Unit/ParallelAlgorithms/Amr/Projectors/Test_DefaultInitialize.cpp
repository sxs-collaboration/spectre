// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

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
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct DefaultTagInt : db::SimpleTag {
  using type = int;
};

struct DefaultTagDouble : db::SimpleTag {
  using type = double;
};

void test_p_refinement() {
  const ElementId<1> element_id{0};
  Element<1> element{element_id, DirectionMap<1, Neighbors<1>>{}};
  const Mesh<1> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  auto box =
      db::create<db::AddSimpleTags<DefaultTagInt, DefaultTagDouble>>(3, 4.2);
  db::mutate_apply<
      amr::projectors::DefaultInitialize<DefaultTagInt, DefaultTagDouble>>(
      make_not_null(&box), std::make_pair(mesh, element));
  CHECK(db::get<DefaultTagInt>(box) == int{});
  CHECK(db::get<DefaultTagDouble>(box) == double{});
}

void test_h_refinement() {
  auto box = db::DataBox<tmpl::list<DefaultTagInt, DefaultTagDouble>>{};
  char unused_extra_arg{'Y'};
  db::mutate_apply<amr::projectors::DefaultInitialize<
      tmpl::list<DefaultTagInt, DefaultTagDouble>>>(make_not_null(&box),
                                                    unused_extra_arg);
  CHECK(db::get<DefaultTagInt>(box) == int{});
  CHECK(db::get<DefaultTagDouble>(box) == double{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Projectors.DefaultInitialize", "[Domain][Unit]") {
  static_assert(
      tt::assert_conforms_to_v<amr::projectors::DefaultInitialize<
                                   tmpl::list<DefaultTagInt, DefaultTagDouble>>,
                               amr::protocols::Projector>);
  test_p_refinement();
  test_h_refinement();
}
