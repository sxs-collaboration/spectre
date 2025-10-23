// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>

#include "Domain/Structure/ElementId.hpp"
#include "Parallel/LockfreeUnorderedSet.hpp"

namespace {
template <bool TrackBucketSize>
void test() {
  CAPTURE(TrackBucketSize);
  {
    Parallel::lockfree::FixedSizeUnorderedSet<
        std::uint64_t,
        Parallel::lockfree::FixedSizeUnorderedSet<
            std::uint64_t>::default_empty_slot_value,
        TrackBucketSize>
        set{8};
    CHECK(set.insert(8));
    CHECK(set.insert(16));
    CHECK(set.erase(8));
    CHECK(set.insert(16));
    CHECK(set.erase(16));
    CHECK_FALSE(set.contains(16));
  }

  const size_t number_of_buckets = 32;
  Parallel::lockfree::FixedSizeUnorderedSet<
      ElementId<3>,
      Parallel::lockfree::FixedSizeUnorderedSet<
          std::uint64_t>::default_empty_slot_value,
      TrackBucketSize>
      set{number_of_buckets};

  CHECK(set.number_of_buckets() == number_of_buckets);
  CHECK(set.insert(ElementId<3>{0}));
  CHECK(set.number_of_buckets() == number_of_buckets);
  CHECK(set.insert(ElementId<3>{1}));
  CHECK(set.number_of_buckets() == number_of_buckets);
  CHECK(set.insert(ElementId<3>{3}));
  CHECK(set.number_of_buckets() == number_of_buckets);
  CHECK(set.insert(ElementId<3>{3}));
  CHECK(set.number_of_buckets() == number_of_buckets);
  CHECK(set.contains(ElementId<3>{3}));
  CHECK(set.contains(ElementId<3>{0}));
  CHECK_FALSE(set.contains(ElementId<3>{7}));
  CHECK(set.contains(ElementId<3>{1}));

  constexpr size_t last_element = TrackBucketSize ? 195 : 229;

  // As of the writing of this test, we need to insert
  // i+5 (i\in[0, last_element)) IDs before we fill the set so full that a
  // collision occurs.
  for (size_t i = 0; i < last_element; ++i) {
    CAPTURE(i);
    CHECK(set.approximate_size() == i + 3);
    CHECK(set.insert(ElementId<3>{i + 5}));
    CHECK(set.approximate_size() == i + 4);
    CHECK(set.number_of_buckets() == number_of_buckets);
    CHECK(set.approximate_load_factor() ==
          approx(static_cast<double>(set.approximate_size()) /
                 static_cast<double>(number_of_buckets)));
  }

  CHECK_FALSE(set.insert(ElementId<3>{last_element + 5}));
  CHECK_FALSE(set.erase(ElementId<3>{last_element + 5}));

  std::vector<ElementId<3>> ids{};
  ids.reserve(set.approximate_size());
  ids.emplace_back(0);
  REQUIRE(set.contains(ids.back()));
  ids.emplace_back(1);
  REQUIRE(set.contains(ids.back()));
  ids.emplace_back(3);
  REQUIRE(set.contains(ids.back()));
  for (size_t i = 0; i < last_element; ++i) {
    ids.emplace_back(i + 5);
    REQUIRE(set.contains(ids.back()));
  }

  std::random_device rd;   // obtain a random seed
  std::mt19937 gen(rd());  // Mersenne Twister engine
  std::uniform_int_distribution dist(static_cast<size_t>(0), ids.size());
  std::vector<ElementId<3>> erased_ids{};
  erased_ids.reserve(ids.size());
  while (set.approximate_size() != 0) {
    // Remove random values that we know are in the set.
    size_t index = dist(gen);
    if (ids.size() == 1) {
      index = 0;
    } else if (index >= ids.size()) {
      index = index % (ids.size() - 1);
    }
    CAPTURE(index);
    CAPTURE(ids.size());
    CAPTURE(ids.at(index));
    erased_ids.push_back(ids.at(index));
    CHECK(set.erase(ids.at(index)));
    ids.erase(std::next(ids.begin(), static_cast<std::ptrdiff_t>(index)));
    CHECK(set.approximate_size() == ids.size());
    CHECK(set.number_of_buckets() == number_of_buckets);
    CHECK(set.approximate_load_factor() ==
          approx(static_cast<double>(set.approximate_size()) /
                 static_cast<double>(number_of_buckets)));
    for (const auto& id : ids) {
      CAPTURE(id);
      CHECK(set.contains(id));
    }
    for (const auto& id : erased_ids) {
      CAPTURE(id);
      CHECK_FALSE(set.contains(id));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.LockfreeUnorderedSet", "[Parallel][Unit]") {
  test<true>();
  test<false>();
}
