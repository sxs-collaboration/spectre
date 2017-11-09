// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t N>
struct NoCopy {
  NoCopy() = default;
  NoCopy(const NoCopy&) = delete;
  NoCopy(NoCopy&&) = default;
  NoCopy& operator=(const NoCopy&) = delete;
  NoCopy& operator=(NoCopy&&) = default;
  ~NoCopy() = default;
};

namespace TestTags {
struct Int : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int";
  using type = int;
};

struct Double : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Double";
  using type = double;
};

template <size_t N>
struct NoCopy : db::DataBoxTag {
  static constexpr db::DataBoxString label = "NoCopy";
  using type = ::NoCopy<N>;
};

template <typename Tag>
struct Negate : db::DataBoxPrefix, db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Negate";
  using tag = Tag;
  static constexpr auto function(const db::item_type<Tag>& x) noexcept {
    return -x;
  }
  using argument_tags = tmpl::list<Tag>;
};

struct AddThree : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "AddThree";
  static constexpr auto function(const int x) noexcept { return x + 3; }
  using argument_tags = tmpl::list<Int>;
  using volume_tags = tmpl::list<Int>;
};

template <size_t VolumeDim>
struct ComplexComputeItem : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ComplexComputeItem";
  static constexpr auto function(const int i, const double d,
                                 const ::NoCopy<1>& /*unused*/,
                                 const ::NoCopy<2>& /*unused*/) noexcept {
    return std::make_pair(i, d);
  }
  using argument_tags = tmpl::list<Int, Double, NoCopy<1>, NoCopy<2>>;
  using volume_tags = tmpl::list<Int, NoCopy<1>>;
};

template <typename>
struct TemplatedDirections : db::DataBoxTag {
  static constexpr db::DataBoxString label = "TemplatedDirections";
  using type = std::unordered_set<Direction<3>>;
};
}  // namespace TestTags
}  // namespace

namespace Tags {
template <typename DirectionsTag>
struct Interface<DirectionsTag, TestTags::Int>
    : InterfaceBase<DirectionsTag, TestTags::Int, TestTags::AddThree> {};
}  // namespace Tags

SPECTRE_TEST_CASE("Unit.Domain.InterfaceItems", "[Unit][Domain]") {
  constexpr size_t dim = 3;
  using internal_directions = Tags::InternalDirections<dim>;
  using templated_directions = TestTags::TemplatedDirections<int>;

  Element<dim> element{ElementId<3>(0),
                       {{Direction<dim>::lower_xi(), {}},
                        {Direction<dim>::upper_xi(), {}},
                        {Direction<dim>::upper_zeta(), {}}}};

  std::unordered_map<Direction<dim>, NoCopy<2>> nocopy_map_item;
  nocopy_map_item.emplace(Direction<dim>::lower_xi(), NoCopy<2>{});
  nocopy_map_item.emplace(Direction<dim>::upper_xi(), NoCopy<2>{});
  nocopy_map_item.emplace(Direction<dim>::upper_zeta(), NoCopy<2>{});
  const auto box = db::create<
      db::AddTags<Tags::Element<dim>,
                  TestTags::Int,
                  Tags::Interface<internal_directions, TestTags::Double>,
                  TestTags::NoCopy<1>,
                  Tags::Interface<internal_directions, TestTags::NoCopy<2>>,
                  templated_directions,
                  Tags::Interface<templated_directions, TestTags::Double>>,
      db::AddComputeItemsTags<
          internal_directions,
          Tags::Interface<internal_directions, Tags::Direction<dim>>,
          TestTags::Negate<TestTags::Int>,
          Tags::Interface<internal_directions, TestTags::Int>,
          Tags::Interface<internal_directions,
                          TestTags::Negate<TestTags::Double>>,
          Tags::Interface<internal_directions,
                          TestTags::ComplexComputeItem<dim>>,
          Tags::Interface<templated_directions, Tags::Direction<dim>>,
          Tags::Interface<templated_directions,
                          TestTags::Negate<TestTags::Double>>>>(
      std::move(element), 5,
      std::unordered_map<Direction<dim>, double>{
          {Direction<dim>::lower_xi(), 1.5},
          {Direction<dim>::upper_xi(), 2.5},
          {Direction<dim>::upper_zeta(), 3.5}},
      NoCopy<1>{}, std::move(nocopy_map_item),
      std::unordered_set<Direction<dim>>{Direction<dim>::upper_xi()},
      std::unordered_map<Direction<dim>, double>{
          {Direction<dim>::upper_xi(), 4.5}});

  CHECK(
      (get<Tags::Interface<internal_directions, Tags::Direction<dim>>>(box)) ==
      (std::unordered_map<Direction<dim>, Direction<dim>>{
          {Direction<dim>::lower_xi(), Direction<dim>::lower_xi()},
          {Direction<dim>::upper_xi(), Direction<dim>::upper_xi()},
          {Direction<dim>::upper_zeta(), Direction<dim>::upper_zeta()}}));

  CHECK(get<TestTags::Negate<TestTags::Int>>(box) == -5);
  CHECK((get<Tags::Interface<internal_directions,
                             TestTags::Negate<TestTags::Double>>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::lower_xi(), -1.5},
            {Direction<dim>::upper_xi(), -2.5},
            {Direction<dim>::upper_zeta(), -3.5}}));

  CHECK((get<Tags::Interface<internal_directions, TestTags::Int>>(box)) ==
        (std::unordered_map<Direction<dim>, int>{
            {Direction<dim>::lower_xi(), 8},
            {Direction<dim>::upper_xi(), 8},
            {Direction<dim>::upper_zeta(), 8}}));

  CHECK((get<Tags::Interface<internal_directions,
                             TestTags::ComplexComputeItem<dim>>>(box)) ==
        (std::unordered_map<Direction<dim>, std::pair<int, double>>{
            {Direction<dim>::lower_xi(), {5, 1.5}},
            {Direction<dim>::upper_xi(), {5, 2.5}},
            {Direction<dim>::upper_zeta(), {5, 3.5}}}));

  CHECK((get<Tags::Interface<templated_directions,
                             TestTags::Negate<TestTags::Double>>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::upper_xi(), -4.5}}));
}
