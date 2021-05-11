// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <unordered_map>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// [inbox_example]
template <size_t Dim>
struct InboxTag {
  using type = std::unordered_map<int, dg::MortarMap<Dim, double>>;
};
// [inbox_example]

template <size_t Dim>
using InboxType = tmpl::type_from<InboxTag<Dim>>;

template <size_t Dim>
void test_has_received_from_all_mortars() {
  CAPTURE(Dim);
  const int time = 0;
  const ElementId<Dim> self_id{1};
  {
    INFO("No neighbors");
    const Element<Dim> element{self_id, {}};
    const tuples::TaggedTuple<InboxTag<Dim>> inbox{InboxType<Dim>{{time, {}}}};
    CHECK(
        dg::has_received_from_all_mortars<InboxTag<Dim>>(time, element, inbox));
  }
  const ElementId<Dim> left_id{0};
  const ElementId<Dim> right_id{2};
  const Element<Dim> element{self_id,
                             {{Direction<Dim>::lower_xi(), {{left_id}, {}}},
                              {Direction<Dim>::upper_xi(), {{right_id}, {}}}}};
  {
    INFO("Complete data");
    const tuples::TaggedTuple<InboxTag<Dim>> inbox{
        InboxType<Dim>{{time,
                        {{{Direction<Dim>::lower_xi(), left_id}, 2.},
                         {{Direction<Dim>::upper_xi(), right_id}, 3.}}}}};
    CHECK(
        dg::has_received_from_all_mortars<InboxTag<Dim>>(time, element, inbox));
  }
  {
    INFO("Missing neighbor data");
    const tuples::TaggedTuple<InboxTag<Dim>> inbox{
        InboxType<Dim>{{time, {{{Direction<Dim>::lower_xi(), left_id}, 2.}}}}};
    CHECK_FALSE(
        dg::has_received_from_all_mortars<InboxTag<Dim>>(time, element, inbox));
  }
  {
    INFO("Missing temporal data");
    const tuples::TaggedTuple<InboxTag<Dim>> inbox{InboxType<Dim>{
        {time + 1, {{{Direction<Dim>::lower_xi(), left_id}, 2.}}}}};
    CHECK_FALSE(
        dg::has_received_from_all_mortars<InboxTag<Dim>>(time, element, inbox));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.HasReceivedFromAllMortars",
                  "[Unit][NumericalAlgorithms]") {
  test_has_received_from_all_mortars<1>();
  test_has_received_from_all_mortars<2>();
  test_has_received_from_all_mortars<3>();
}
