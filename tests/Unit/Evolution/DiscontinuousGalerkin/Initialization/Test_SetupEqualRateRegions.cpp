// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.tpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SetupEqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace {
class TestRegions {
 public:
  using creation_tags = tmpl::list<>;

  TestRegions() = default;

  static std::unordered_map<std::string, size_t> regions() {
    std::unordered_map<std::string, size_t> result{};
    result.emplace("Region1", 1);
    result.emplace("Region2", 2);
    return result;
  }

  template <size_t Dim>
  static bool is_in_region(const size_t region, const ElementId<Dim>& element) {
    // Regions for the test are block1 and block2+block3.  We add
    // block 5 to both to test an error.
    if (region == 1) {
      return element.block_id() == 1 or element.block_id() == 5;
    } else if (region == 2) {
      return element.block_id() == 2 or element.block_id() == 3 or
             element.block_id() == 5;
    }

    ERROR("Invalid region");
  }

  void pup(PUP::er&);  // unused
};

enum class Nonconforming { Conforming, Large, Small, LargeError };

template <size_t Dim>
void test() {
  for (size_t block = 0; block < 6; ++block) {
    for (const auto conforming :
         {Nonconforming::Conforming, Nonconforming::Large, Nonconforming::Small,
          Nonconforming::LargeError}) {
      if (Dim == 1 and conforming != Nonconforming::Conforming) {
        // Can't have nonconforming boundaries in 1D.
        continue;
      }
      if (conforming == Nonconforming::LargeError and block != 2) {
        // Needs the next block to be in the same region to trigger
        // error.
        continue;
      }

      const size_t lts_ratio = 32;

      std::optional<size_t> expected_ratio{};
      std::unordered_map<Side, evolution::dg::TimeSteppingPolicy>
          expected_policy{};
      expected_policy[Side::Upper] =
          evolution::dg::TimeSteppingPolicy::Conservative;
      expected_policy[Side::Lower] =
          evolution::dg::TimeSteppingPolicy::Conservative;

      // For the test, block 1 is a region, and blocks 2 and 3 are a
      // second region.  The upper side is internal to the block and
      // the lower side goes to the next block.
      if (block == 1 or block == 3) {
        expected_ratio.emplace(lts_ratio);
        expected_policy[Side::Upper] =
            evolution::dg::TimeSteppingPolicy::EqualRate;
      } else if (block == 2) {
        expected_ratio.emplace(lts_ratio);
        expected_policy[Side::Upper] =
            evolution::dg::TimeSteppingPolicy::EqualRate;
        expected_policy[Side::Lower] =
            evolution::dg::TimeSteppingPolicy::EqualRate;
      }

      auto lower_segments = make_array<Dim>(SegmentId(0, 0));
      lower_segments[0] = SegmentId(1, 0);
      auto upper_segments = make_array<Dim>(SegmentId(0, 0));
      upper_segments[0] = SegmentId(1, 1);

      const ElementId<Dim> element_id(block, lower_segments);

      DirectionMap<Dim, Neighbors<Dim>> initialize_neighbors{};
      {
        const auto direction = Direction<Dim>::upper_xi();
        std::unordered_set<ElementId<Dim>> neighbor_ids{};
        neighbor_ids.emplace(block, upper_segments);
        initialize_neighbors.emplace(
            direction, Neighbors<Dim>(neighbor_ids,
                                      OrientationMap<Dim>::create_aligned()));
      }
      {
        const auto direction = Direction<Dim>::lower_xi();
        std::unordered_set<ElementId<Dim>> neighbor_ids{};
        std::unordered_map<size_t, OrientationMap<Dim>> orientations{};
        orientations.emplace(block + 1, OrientationMap<Dim>::create_aligned());
        if (Dim == 1 or conforming == Nonconforming::Small) {
          neighbor_ids.emplace(block + 1, upper_segments);
        } else if (conforming != Nonconforming::LargeError) {
          auto split_segments = upper_segments;
          gsl::at(split_segments, 1) = SegmentId(1, 0);
          neighbor_ids.emplace(block + 1, split_segments);
          gsl::at(split_segments, 1) = SegmentId(1, 1);
          neighbor_ids.emplace(block + 1, split_segments);
        } else {
          neighbor_ids.emplace(block + 1, upper_segments);
          neighbor_ids.emplace(0, upper_segments);
          orientations.emplace(0, OrientationMap<Dim>::create_aligned());
        }
        initialize_neighbors.emplace(
            direction, Neighbors<Dim>(neighbor_ids, orientations,
                                      conforming == Nonconforming::Conforming));
      }

      // NOLINTNEXTLINE(misc-const-correctness) - is moved
      Element<Dim> element(element_id, std::move(initialize_neighbors));

      DirectionalIdMap<Dim, evolution::dg::MortarInfo<Dim>>
          initial_mortar_infos{};
      for (const auto& [direction, neighbors] : element.neighbors()) {
        if (direction == Direction<Dim>::lower_xi() or
            conforming == Nonconforming::Conforming or
            conforming == Nonconforming::Small) {
          for (const auto& neighbor : neighbors) {
            initial_mortar_infos[{direction, neighbor}].time_stepping_policy() =
                evolution::dg::TimeSteppingPolicy::Conservative;
          }
        } else {
          initial_mortar_infos[{direction, element_id}].time_stepping_policy() =
              evolution::dg::TimeSteppingPolicy::Conservative;
        }
      }
      const auto mortar_info_size = initial_mortar_infos.size();

      const Slab slab(3.7, 19.2);
      const TimeDelta time_step = slab.duration() / lts_ratio;

      auto box = db::create<
          db::AddSimpleTags<Tags::FixedLtsRatio,
                            evolution::dg::Tags::MortarInfo<Dim>,
                            domain::Tags::Element<Dim>,
                            evolution::dg::Tags::ConcreteEqualRateRegions<
                                Dim, tmpl::list<TestRegions>>,
                            Tags::TimeStep>,
          db::AddComputeTags<evolution::dg::Tags::EqualRateRegionsRef<
              Dim, tmpl::list<TestRegions>>>>(
          std::optional<size_t>{}, std::move(initial_mortar_infos),
          std::move(element),
          evolution::dg::EqualRateRegions<Dim, tmpl::list<TestRegions>>{},
          time_step);

      if (block == 5) {
        // TestRegions intentionally gives invalid results for block 5.
        CHECK_THROWS_WITH(
            db::mutate_apply<
                evolution::dg::Initialization::SetupLocalEqualRateRegion<Dim>>(
                make_not_null(&box)),
            Catch::Matchers::ContainsSubstring(
                "in multiple equal-rate regions"));
        continue;
      } else if (conforming == Nonconforming::LargeError) {
        CHECK_THROWS_WITH(
            db::mutate_apply<
                evolution::dg::Initialization::SetupLocalEqualRateRegion<Dim>>(
                make_not_null(&box)),
            Catch::Matchers::ContainsSubstring(
                "only some neighbors across mortar"));
        continue;
      }
      db::mutate_apply<
          evolution::dg::Initialization::SetupLocalEqualRateRegion<Dim>>(
          make_not_null(&box));

      CHECK(db::get<Tags::FixedLtsRatio>(box) == expected_ratio);

      const auto& mortar_infos =
          db::get<evolution::dg::Tags::MortarInfo<Dim>>(box);
      CHECK(mortar_infos.size() == mortar_info_size);
      for (const auto& [mortar_id, info] : mortar_infos) {
        CHECK(info.time_stepping_policy() ==
              expected_policy.at(mortar_id.direction().side()));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.Initialization.SetupEqualRateRegions",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
