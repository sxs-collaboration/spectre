// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/MaybeOwned.hpp"
#include "Utilities/Gsl.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.DataStructures.MaybeOwned", "[Unit][DataStructures]") {
  // default constructor
  {
    MaybeOwned<std::unique_ptr<int>> defaulted{};
    CHECK(defaulted.is_owning());
    CHECK(*defaulted == nullptr);
  }

  // simple case
  {
    int value = 5;
    MaybeOwned unowned = make_not_null(&value);
    MaybeOwned owned = value;
    MaybeOwned owned2 = 6;

    static_assert(std::is_same_v<decltype(unowned)::value_type, int>);
    static_assert(std::is_same_v<decltype(owned)::value_type, int>);
    static_assert(std::is_same_v<decltype(owned2)::value_type, int>);

    CHECK(not unowned.is_owning());
    CHECK(owned.is_owning());
    CHECK(owned2.is_owning());

    CHECK(*unowned == 5);
    CHECK(*owned == 5);
    CHECK(*owned2 == 6);

    CHECK(&*unowned == &value);
    CHECK(&*owned != &value);

    *owned = 7;
    CHECK(*owned == 7);
    *unowned = 8;
    CHECK(value == 8);
  }

  // non-copyable and const argument
  {
    const auto value = std::make_unique<int>(5);
    const MaybeOwned unowned = make_not_null(&value);
    const MaybeOwned owned = std::make_unique<int>(6);

    static_assert(std::is_same_v<decltype(unowned)::value_type,
                                 const std::unique_ptr<int>>);
    // Owning versions can't deduce constness.
    static_assert(
        std::is_same_v<decltype(owned)::value_type, std::unique_ptr<int>>);

    CHECK(not unowned.is_owning());
    CHECK(owned.is_owning());

    CHECK(**unowned == 5);
    CHECK(**owned == 6);

    CHECK(&*unowned == &value);
    CHECK(&*owned != &value);

    CHECK(unowned->get() == (*unowned).get());
    CHECK(owned->get() == (*owned).get());
  }

  // Moving
  {
    MaybeOwned owned(std::make_unique<int>(5));
    MaybeOwned move_owned = std::move(owned);
    CHECK(owned.is_owning());  // NOLINT(bugprone-use-after-move)
    CHECK(move_owned.is_owning());
    CHECK(&*move_owned != &*owned);
    CHECK(*owned == nullptr);
    CHECK(**move_owned == 5);
    MaybeOwned<std::unique_ptr<int>> move_owned2{};
    move_owned2 = std::move(move_owned);
    CHECK(move_owned.is_owning());  // NOLINT(bugprone-use-after-move)
    CHECK(move_owned2.is_owning());
    CHECK(&*move_owned2 != &*move_owned);
    CHECK(*move_owned == nullptr);
    CHECK(**move_owned2 == 5);
  }
  {
    auto value = std::make_unique<int>(5);
    MaybeOwned unowned = make_not_null(&value);
    MaybeOwned move_unowned = std::move(unowned);
    CHECK(not move_unowned.is_owning());
    CHECK(&*move_unowned == &value);
    MaybeOwned<std::unique_ptr<int>> move_unowned2{};
    move_unowned2 = std::move(move_unowned);
    CHECK(not move_unowned2.is_owning());
    CHECK(&*move_unowned2 == &value);
  }

  // Conversion to const
  {
    auto value = std::make_unique<int>(5);
    MaybeOwned<std::unique_ptr<int>> unowned = make_not_null(&value);
    MaybeOwned<std::unique_ptr<int>> owned = std::make_unique<int>(5);
    const MaybeOwned<const std::unique_ptr<int>> const_unowned =
        std::move(unowned);
    const MaybeOwned<const std::unique_ptr<int>> const_owned = std::move(owned);
    CHECK(not const_unowned.is_owning());
    CHECK(const_owned.is_owning());
    CHECK(**const_unowned == 5);
    CHECK(**const_owned == 5);
  }
  {
    auto value = std::make_unique<int>(5);
    MaybeOwned<std::unique_ptr<int>> unowned = make_not_null(&value);
    MaybeOwned<std::unique_ptr<int>> owned = std::make_unique<int>(5);
    // Set up in incorrect state to be overwritten.
    MaybeOwned<const std::unique_ptr<int>> const_unowned =
        std::make_unique<int>(5);
    const_unowned = std::move(unowned);
    MaybeOwned<const std::unique_ptr<int>> const_owned = make_not_null(&value);
    const_owned = std::move(owned);
    CHECK(not const_unowned.is_owning());
    CHECK(const_owned.is_owning());
    CHECK(**const_unowned == 5);
    CHECK(**const_owned == 5);
  }

  // Assignment of values
  {
    const auto value = std::make_unique<int>(5);
    const auto value2 = std::make_unique<int>(6);
    MaybeOwned<const std::unique_ptr<int>> maybe = make_not_null(&value);
    CHECK(not maybe.is_owning());
    CHECK(**maybe == 5);
    // unowning -> unowning
    maybe = make_not_null(&value2);
    CHECK(not maybe.is_owning());
    CHECK(**maybe == 6);
    // unowning -> owning
    maybe = std::make_unique<int>(7);
    CHECK(maybe.is_owning());
    CHECK(**maybe == 7);
    // owning -> owning
    maybe = std::make_unique<int>(8);
    CHECK(maybe.is_owning());
    CHECK(**maybe == 8);
    // owning -> unowning
    maybe = &value;
    CHECK(not maybe.is_owning());
    CHECK(**maybe == 5);

    auto non_const = std::make_unique<int>(9);
    maybe = &non_const;
    CHECK(not maybe.is_owning());
    CHECK(**maybe == 9);
  }

  // to_owned()
  {
    std::vector<int> unowned_source{1, 2, 3};
    const int* const unowned_data = unowned_source.data();
    std::vector<int> owned_source{1, 2, 3};
    const int* const owned_data = owned_source.data();
    std::vector<int> owned_source2{1, 2, 3};
    const int* const owned_data2 = owned_source2.data();
    MaybeOwned<std::vector<int>> non_owning = make_not_null(&unowned_source);
    MaybeOwned<std::vector<int>> owning = std::move(owned_source);
    MaybeOwned<const std::vector<int>> const_non_owning =
        make_not_null(&unowned_source);
    MaybeOwned<const std::vector<int>> const_owning = std::move(owned_source2);
    CHECK(non_owning->data() == unowned_data);
    CHECK(owning->data() == owned_data);
    CHECK(const_non_owning->data() == unowned_data);
    CHECK(const_owning->data() == owned_data2);
    const auto taken_non_owning = std::move(non_owning).to_owned();
    const auto taken_owning = std::move(owning).to_owned();
    const auto taken_const_non_owning = std::move(const_non_owning).to_owned();
    const auto taken_const_owning = std::move(const_owning).to_owned();
    CHECK(taken_non_owning.data() != unowned_data);
    CHECK(taken_owning.data() == owned_data);
    CHECK(taken_const_non_owning.data() != unowned_data);
    CHECK(taken_const_owning.data() == owned_data2);
    CHECK(unowned_source.data() == unowned_data);
  }
}
}  // namespace
