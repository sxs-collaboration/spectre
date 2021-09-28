// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace db::detail {

// Used to label the different types of items in a DataBox (which are described
// in detail below for each specialization of Item).
enum class ItemType { Mutable, Compute, Reference, Invalid };

// A unique item in a DataBox labeled by Tag
template <typename Tag, ItemType = db::is_mutable_item_tag_v<Tag>
                                       ? ItemType::Mutable
                                       : (db::is_compute_tag_v<Tag>
                                              ? ItemType::Compute
                                              : (db::is_reference_tag_v<Tag>
                                                     ? ItemType::Reference
                                                     : ItemType::Invalid))>
class Item {
  static_assert(
      db::is_non_base_tag_v<Tag>,
      "The Tag of an Item in the Databox must be derived from db::SimpleTag");
};

// A mutable item in a DataBox
//
// A mutable item is an item in a DataBox that is initialized when the DataBox
// is constructed using either db::create or db::create_from
//
// Its value may be fetched by calling db::get (which calls get)
//
// Its value may be changed by calling db::mutate (which calls mutate)
template <typename Tag>
class Item<Tag, ItemType::Mutable> {
 public:
  static constexpr ItemType item_type = ItemType::Mutable;

  using value_type = typename Tag::type;

  constexpr Item() = default;
  constexpr Item(Item const&) = default;
  constexpr Item(Item&&) = default;
  constexpr Item& operator=(Item const&) = default;
  constexpr Item& operator=(Item&&) = default;
  ~Item() = default;

  explicit Item(value_type value) : value_(std::move(value)) {}

  const value_type& get() const { return value_; }

  value_type& mutate() { return value_; }

 private:
  value_type value_{};
};

// A compute item in a DataBox
//
// A compute item is an item in a DataBox whose value depends upon other items
// in the DataBox.  It is lazily evaluated (i.e. not computed until it is
// retrieved) and will be reevaluated if any of the items upon which it depends
// is changed.
//
// A compute item is default constructed when the DataBox is constructed.
//
// When a compute item is fetched via db::get, db::get will check whether it has
// already been evaluated (by calling evaluated).  If it has been evaluated,
// db::get fetches its value (via get).  If it has not been evaluated (either
// initially or after reset has been called), db::get will call evaluate which
// will compute the value using tag::function.
//
// When db::mutate is called on a mutable item, all compute items that depend
// (directly or indirectly) on the mutated item will have their reset function
// called.
//
// A compute item may not be directly mutated (its value only changes after one
// of its dependencies changes and it is fetched again)
template <typename Tag>
class Item<Tag, ItemType::Compute> {
 public:
  static constexpr ItemType item_type = ItemType::Compute;

  using value_type = typename Tag::type;

  constexpr Item() = default;
  constexpr Item(Item const&) = default;
  constexpr Item(Item&&) = default;
  constexpr Item& operator=(Item const&) = default;
  constexpr Item& operator=(Item&&) = default;
  ~Item() = default;

  const value_type& get() const { return value_; }

  bool evaluated() const { return evaluated_; }

  void reset() { evaluated_ = false; }

  template <typename... Args>
  void evaluate(const Args&... args) const {
    Tag::function(make_not_null(&value_), args...);
    evaluated_ = true;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | evaluated_;
    if (evaluated_) {
      p | value_;
    }
  }

 private:
  mutable value_type value_{};
  mutable bool evaluated_{false};
};

// A reference item in the DataBox
//
// A reference item is an item in a DataBox used as a const reference to a
// subitem of another item (called the parent item) contained in the DataBox
//
// Its value may be fetched via db::get (by calling Tag::get directly)
//
// A reference item cannot be used to mutate its value.
template <typename Tag>
class Item<Tag, ItemType::Reference> {
 public:
  static constexpr ItemType item_type = ItemType::Reference;

  constexpr Item() = default;
  constexpr Item(Item const&) = default;
  constexpr Item(Item&&) = default;
  constexpr Item& operator=(Item const&) = default;
  constexpr Item& operator=(Item&&) = default;
  ~Item() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
}  // namespace db::detail
/// \endcond
