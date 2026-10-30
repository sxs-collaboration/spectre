// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <type_traits>
#include <utility>

#include "Utilities/Gsl.hpp"

/// An object or reference.
///
/// This class can either store an object or point to a preexisting
/// one, giving uniform access to the two cases.
template <typename T>
class MaybeOwned {
 public:
  using value_type = T;

  MaybeOwned(MaybeOwned&&) = default;
  MaybeOwned& operator=(MaybeOwned&&) = default;

  // Not clear what correct semantics for copying a non-owning object
  // would be.
  MaybeOwned(const MaybeOwned&) = delete;
  MaybeOwned& operator=(const MaybeOwned&) = delete;

  ~MaybeOwned() = default;

  /// A default constructed MaybeOwned<T> owns a default-constructed T.
  MaybeOwned() : owned_value_(std::in_place) {}

  /// Create an owning object.
  // NOLINTNEXTLINE(google-explicit-constructor)
  MaybeOwned(std::remove_const_t<T> owned) : owned_value_(std::move(owned)) {}
  /// Create a non-owning object.
  // NOLINTNEXTLINE(google-explicit-constructor)
  MaybeOwned(const gsl::not_null<T*> unowned) : unowned_value_(unowned.get()) {}

  // The conversion sequence
  //   not_null<T*> -> not_null<const T*> -> MaybeOwned<const T>
  // is too complex to be found automatically.

  /// Create a non-owning object when the previous constructor needs
  /// too many implicit conversions.
  // NOLINTNEXTLINE(google-explicit-constructor)
  MaybeOwned(const gsl::not_null<std::remove_const_t<T>*> unowned)
    requires std::is_const_v<T>
      : MaybeOwned(gsl::not_null<T*>(unowned)) {}

  /// Conversion from MaybeOwned<T> to MaybeOwned<const T>
  // NOLINTNEXTLINE(google-explicit-constructor)
  MaybeOwned(MaybeOwned<std::remove_const_t<T>>&& other)
    requires std::is_const_v<T>
      : owned_value_(std::move(other).owned_value_),
        unowned_value_(other.unowned_value_) {}

  /// Conversion from MaybeOwned<T> to MaybeOwned<const T>
  MaybeOwned& operator=(MaybeOwned<std::remove_const_t<T>>&& other)
    requires std::is_const_v<T>
  {
    owned_value_ = std::move(other).owned_value_;
    unowned_value_ = other.unowned_value_;
    return *this;
  }

  /// Change the object to owning, destroying any previously owned value.
  MaybeOwned& operator=(std::remove_const_t<T> owned) {
    owned_value_.emplace(std::move(owned));
    unowned_value_ = nullptr;
    return *this;
  }
  /// Change the object to non-owning.
  MaybeOwned& operator=(const gsl::not_null<T*> unowned) {
    owned_value_.reset();
    unowned_value_ = unowned.get();
    return *this;
  }

  /// Access the contained or referenced object
  /// @{
  T& operator*() {
    return owned_value_.has_value() ? *owned_value_ : *unowned_value_;
  }
  const T& operator*() const {
    return owned_value_.has_value() ? *owned_value_ : *unowned_value_;
  }

  T* operator->() { return &**this; }
  const T* operator->() const { return &**this; }
  /// @}

  bool is_owning() const { return owned_value_.has_value(); }

  /// Get the owned object, or a copy if non-owning.
  std::remove_const_t<T> to_owned() && {
    if (owned_value_.has_value()) {
      return std::move(*owned_value_);
    } else {
      return *unowned_value_;
    }
  }

 private:
  friend class MaybeOwned<const T>;

  std::optional<std::remove_const_t<T>> owned_value_{};
  T* unowned_value_ = nullptr;
};

template <typename T>
MaybeOwned(T) -> MaybeOwned<T>;

// Storing a reference or pointer to a not_null seems unlikely enough
// that we assume the user wanted the other behavior.  The class can
// still be instantiated with an explicit argument if someone really
// wants that.
template <typename T>
MaybeOwned(gsl::not_null<T*>) -> MaybeOwned<T>;

// Disable pointers, as the user probably meant to use the not_null
// call, but not definitely enough that inferring seems like a good
// idea.
//
// "= delete" would be more appropriate, but the standards committee
// appears to have approved that and then just forgotten to add it to
// the standard or something.
template <typename T>
MaybeOwned(T*) -> MaybeOwned<void>;
