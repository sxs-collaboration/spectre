// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>

#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.tpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Block>
struct Option {
  using type = size_t;
};

template <size_t... Blocks>
class BlockRegion {
 public:
  explicit BlockRegion() = default;

  using creation_tags = tmpl::list<Option<Blocks>...>;

  explicit BlockRegion(const typename tmpl::has_type<tmpl::size_t<Blocks>,
                                                     size_t>::type... options) {
    CHECK((... and (Blocks == options)));

    // Set this up in the constructor to make sure the constructor is
    // actually called.
    for (const size_t block : {Blocks...}) {
      regions_.emplace("Block" + std::to_string(block), block);
    }
  }

  std::unordered_map<std::string, size_t> regions() const { return regions_; }

  template <size_t Dim>
  bool is_in_region(const size_t region, const ElementId<Dim>& element) const {
    return element.block_id() == region;
  }

  void pup(PUP::er& p) { p | regions_; }

 private:
  std::unordered_map<std::string, size_t> regions_{};
};

template <typename Regions>
void check_regions(const Regions& regions) {
  const auto& region_map = regions.regions();
  CHECK(region_map.size() == 3);
  REQUIRE(region_map.contains("Block1"));
  REQUIRE(region_map.contains("Block3"));
  REQUIRE(region_map.contains("Block4"));
  const evolution::dg::EqualRateRegionId block1 = region_map.at("Block1");
  const evolution::dg::EqualRateRegionId block3 = region_map.at("Block3");
  const evolution::dg::EqualRateRegionId block4 = region_map.at("Block4");
  test_serialization(block1);
  test_serialization(block3);
  test_serialization(block4);

  const auto& region_name_map = regions.region_names();
  CHECK(region_name_map.size() == 3);
  CHECK(region_name_map.at(block1) == "Block1");
  CHECK(region_name_map.at(block3) == "Block3");
  CHECK(region_name_map.at(block4) == "Block4");

  CHECK(regions.is_in_region(block1, ElementId<1>{1}));
  CHECK(regions.is_in_region(block4, ElementId<1>{4}));
  CHECK(not regions.is_in_region(block3, ElementId<1>{1}));
  CHECK(not regions.is_in_region(block1, ElementId<1>{0}));
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.EqualRateLts.EqualRateRegions",
                  "[Unit][Evolution]") {
  {
    const evolution::dg::EqualRateRegions<
        1, tmpl::list<BlockRegion<1, 3>, BlockRegion<4>>>
        regions{1, 3, 4};
    const auto copied_regions = serialize_and_deserialize(regions);
    check_regions(regions);
    check_regions(copied_regions);
    check_regions<evolution::dg::EqualRateRegionsBase<1>>(regions);
    check_regions<evolution::dg::EqualRateRegionsBase<1>>(copied_regions);
  }

  {
    const evolution::dg::EqualRateRegions<1, tmpl::list<>> empty_regions{};
    const auto copied_empty_regions = serialize_and_deserialize(empty_regions);
    CHECK(empty_regions.regions().empty());
    CHECK(empty_regions.region_names().empty());
    CHECK(copied_empty_regions.regions().empty());
    CHECK(copied_empty_regions.region_names().empty());
    // Not allowed to call other methods as they require valid region ids.
  }

  CHECK_THROWS_WITH(
      (evolution::dg::EqualRateRegions<
          1, tmpl::list<BlockRegion<1, 3>, BlockRegion<3>>>{1, 3, 3}),
      Catch::Matchers::ContainsSubstring(
          "Generated multiple regions named Block3"));

  CHECK(get_output(evolution::dg::EqualRateRegionId{3, 5}) == "{3,5}");
}
}  // namespace
