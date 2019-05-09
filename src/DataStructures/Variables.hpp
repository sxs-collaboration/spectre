// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Variables

#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <ostream>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeSignalingNan.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_forward_declare MakeWithValueImpl
// IWYU pragma: no_forward_declare Variables

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;

template <typename TagsList>
class Variables;
/// \endcond

namespace Tags {
template <typename TagsList>
struct Variables : db::SimpleTag {
  static_assert(tt::is_a<tmpl::list, TagsList>::value,
                "The TagsList passed to Tags::Variables is not a typelist");
  using tags_list = TagsList;
  using type = ::Variables<TagsList>;
  static std::string name() noexcept {
    std::string tag_name{"Variables("};
    size_t iter = 0;
    tmpl::for_each<TagsList>([&tag_name, &iter ](auto tag) noexcept {
      tag_name += tmpl::type_from<decltype(tag)>::name();
      if (iter + 1 != tmpl::size<TagsList>::value) {
        tag_name += ",";
      }
      iter++;
    });
    return tag_name + ")";
  }
};
}  // namespace Tags

/*!
 * \ingroup DataStructuresGroup
 * \brief A Variables holds a contiguous memory block with Tensors pointing
 * into it.
 *
 * The `Tags` are `struct`s that must have a public type alias `type` whose
 * value must be a `Tensor<DataVector, ...>`, a `static' method `name()` that
 * returns a `std::string` of the tag name, and must derive off of
 * `db::SimpleTag`. In general, they should be DataBoxTags that are not compute
 * items. For example,
 *
 * \snippet Test_Variables.cpp simple_variables_tag
 *
 * Prefix tags can also be stored and their format is:
 *
 * \snippet Test_Variables.cpp prefix_variables_tag
 *
 * #### Design Decisions
 *
 * The `Variables` class is designed to hold several different `Tensor`s
 * performing one memory allocation for all the `Tensor`s. The advantage is that
 * memory allocations are quite expensive, especially in a parallel environment.
 *
 * `Variables` stores the data it owns in a `std::unique_ptr<double[],
 * decltype(&free)>` instead of a `std::vector` because allocating the
 * `unique_ptr` with `malloc` allows us to avoid initializing the memory
 * completely in release mode when no value is passed to the constructor.
 * Additionally, if the macro `SPECTRE_NAN_INIT` is defined, initialization with
 * `NaN`s is done even in release mode.
 */
template <typename... Tags>
class Variables<tmpl::list<Tags...>> {
 public:
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  static constexpr auto transpose_flag = blaze::defaultTransposeFlag;

  /// A typelist of the Tags whose variables are held
  using tags_list = tmpl::list<Tags...>;
  static_assert(sizeof...(Tags) > 0,
                "You must provide at least one tag to the Variables "
                "for type inference");

  static_assert(
      (tmpl2::flat_all_v<
           cpp17::is_same_v<typename Tags::type::type,
                            typename tmpl::front<tags_list>::type::type>...> or
       tmpl2::flat_all_v<is_spin_weighted_of_same_type_v<
           typename tmpl::front<tags_list>::type::type,
           typename Tags::type::type>...>),
      "All tensors stored in a single Variables must "
      "have the same internal storage type.");

  static_assert(
      tmpl2::flat_all_v<cpp17::is_same_v<
          cpp17::void_t<typename Tags::type::type::value_type>, void>...>,
      "The tensor stored in a Variables must have as member `type` "
      "a vector with a member `value_type` (e.g. Tensors of doubles are "
      "disallowed, use instead a Tensor of DataVectors).");

  using vector_type = tmpl::conditional_t<
      is_any_spin_weighted_v<typename tmpl::front<tags_list>::type::value_type>,
      typename tmpl::front<tags_list>::type::type::value_type,
      typename tmpl::front<tags_list>::type::type>;
  using value_type = typename vector_type::value_type;
  using allocator_type = std::allocator<value_type>;
  using pointer_type =
      PointerVector<value_type, blaze::unaligned, blaze::unpadded,
                    transpose_flag,
                    blaze::DynamicVector<value_type, transpose_flag>>;

  static_assert(
      cpp17::is_fundamental_v<value_type> or
          tt::is_a_v<std::complex, value_type>,
      "`value_type` of the Variables (so the storage type of the vector type "
      "within the tensors in the Variables) must be either a fundamental type "
      "or a std::complex. If this constraint is relaxed, the value_type "
      "should be handled differently in the Variables, including pass by "
      "reference.");

  /// The number of variables of the Variables object is holding. E.g.
  /// \f$\psi_{ab}\f$ would be counted as one variable.
  static constexpr auto number_of_variables = sizeof...(Tags);

  /// \cond
  // If you encounter an error of the `size()` function not existing you are
  // not filling the Variables with Tensors. Variables can be generalized to
  // holding containers other than Tensor by having the containers have a
  // `size()` function that in most cases should return 1. For Tensors the
  // `size()` function returns the number of independent components.
  template <typename State, typename Element>
  struct number_of_independent_components_helper {
    using type =
        typename tmpl::plus<State, tmpl::int32_t<Element::type::size()>>::type;
  };
  /// \endcond

  /// The total number of independent components of all the variables. E.g.
  /// a rank-2 symmetric spacetime Tensor \f$\psi_{ab}\f$ in 3 spatial
  /// dimensions would have 10 independent components.
  static constexpr size_t number_of_independent_components =
      tmpl::fold<tmpl::list<Tags...>, tmpl::int32_t<0>,
                 number_of_independent_components_helper<
                     tmpl::_state, tmpl::_element>>::value;

  /// Default construct an empty Variables class, Charm++ needs this
  Variables() noexcept;

  explicit Variables(size_t number_of_grid_points) noexcept;

  Variables(size_t number_of_grid_points, value_type value) noexcept;

  Variables(Variables&& rhs) noexcept = default;
  Variables& operator=(Variables&& rhs) noexcept;

  Variables(const Variables& rhs) noexcept;
  Variables& operator=(const Variables& rhs) noexcept;

  // @{
  /// Copy and move semantics for wrapped variables
  template <typename... WrappedTags,
            Requires<tmpl2::flat_all<std::is_same<
                db::remove_all_prefixes<WrappedTags>,
                db::remove_all_prefixes<Tags>>::value...>::value> = nullptr>
  explicit Variables(Variables<tmpl::list<WrappedTags...>>&& rhs) noexcept;
  template <typename... WrappedTags,
            Requires<tmpl2::flat_all<std::is_same<
                db::remove_all_prefixes<WrappedTags>,
                db::remove_all_prefixes<Tags>>::value...>::value> = nullptr>
  Variables& operator=(Variables<tmpl::list<WrappedTags...>>&& rhs) noexcept;

  template <typename... WrappedTags,
            Requires<tmpl2::flat_all<std::is_same<
                db::remove_all_prefixes<WrappedTags>,
                db::remove_all_prefixes<Tags>>::value...>::value> = nullptr>
  explicit Variables(const Variables<tmpl::list<WrappedTags...>>& rhs) noexcept;
  template <typename... WrappedTags,
            Requires<tmpl2::flat_all_v<std::is_same<
                db::remove_all_prefixes<WrappedTags>,
                db::remove_all_prefixes<Tags>>::value...>> = nullptr>
  Variables& operator=(
      const Variables<tmpl::list<WrappedTags...>>& rhs) noexcept;
  // @}

  /// \cond HIDDEN_SYMBOLS
  ~Variables() noexcept = default;
  /// \endcond

  // @{
  /// Initialize a Variables to the state it would have after calling
  /// the constructor with the same arguments.
  // this should be updated if we ever use a variables which has a `value_type`
  // larger than ~2 doubles in size.
  void initialize(size_t number_of_grid_points) noexcept;
  void initialize(size_t number_of_grid_points, value_type value) noexcept;
  // @}

  constexpr SPECTRE_ALWAYS_INLINE size_t number_of_grid_points() const
      noexcept {
    return number_of_grid_points_;
  }

  /// Number of grid points * number of independent components
  constexpr SPECTRE_ALWAYS_INLINE size_type size() const noexcept {
    return size_;
  }

  //{@
  /// Access pointer to underlying data
  value_type* data() noexcept { return variable_data_.data(); }
  const value_type* data() const noexcept { return variable_data_.data(); }
  //@}

  /// \cond HIDDEN_SYMBOLS
  /// Needed because of limitations and inconsistency between compiler
  /// implementations of friend function templates with auto return type of
  /// class templates
  const auto& get_variable_data() const noexcept { return variable_data_; }
  /// \endcond

  // clang-tidy: redundant-declaration
  template <typename Tag, typename TagList>
  friend constexpr typename Tag::type& get(  // NOLINT
      Variables<TagList>& v) noexcept;
  template <typename Tag, typename TagList>
  friend constexpr const typename Tag::type& get(  // NOLINT
      const Variables<TagList>& v) noexcept;

  /// Serialization for Charm++.
  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  // @{
  /// \brief Assign a subset of the `Tensor`s from another Variables or a
  /// tuples::TaggedTuple
  ///
  /// \note There is no need for an rvalue overload because we need to copy into
  /// the contiguous array anyway
  template <typename... SubsetOfTags,
            Requires<tmpl2::flat_all_v<tmpl::list_contains_v<
                tmpl::list<Tags...>, SubsetOfTags>...>> = nullptr>
  void assign_subset(
      const Variables<tmpl::list<SubsetOfTags...>>& vars) noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(
        (get<SubsetOfTags>(*this) = get<SubsetOfTags>(vars)));
  }

  template <typename... SubsetOfTags,
            Requires<tmpl2::flat_all_v<tmpl::list_contains_v<
                tmpl::list<Tags...>, SubsetOfTags>...>> = nullptr>
  void assign_subset(
      const tuples::TaggedTuple<SubsetOfTags...>& vars) noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(
        (get<SubsetOfTags>(*this) = get<SubsetOfTags>(vars)));
  }
  // @}

  /// Create a Variables from a subset of the `Tensor`s in this
  /// Variables
  template <typename SubsetOfTags>
  Variables<SubsetOfTags> extract_subset() const noexcept {
    Variables<SubsetOfTags> sub_vars(number_of_grid_points());
    tmpl::for_each<SubsetOfTags>([&](const auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      get<tag>(sub_vars) = get<tag>(*this);
    });
    return sub_vars;
  }

  /// Converting constructor for an expression to a Variables class
  // clang-tidy: mark as explicit (we want conversion to Variables)
  template <typename VT, bool VF>
  Variables(const blaze::Vector<VT, VF>& expression) noexcept;  // NOLINT

  template <typename VT, bool VF>
  Variables& operator=(const blaze::Vector<VT, VF>& expression) noexcept;

  template <typename... WrappedTags,
            Requires<tmpl2::flat_all_v<
                cpp17::is_same_v<db::remove_all_prefixes<WrappedTags>,
                                 db::remove_all_prefixes<Tags>>...>> = nullptr>
  SPECTRE_ALWAYS_INLINE Variables& operator+=(
      const Variables<tmpl::list<WrappedTags...>>& rhs) noexcept {
    variable_data_ += rhs.variable_data_;
    return *this;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE Variables& operator+=(
      const blaze::Vector<VT, VF>& rhs) noexcept {
    variable_data_ += rhs;
    return *this;
  }

  template <typename... WrappedTags,
            Requires<tmpl2::flat_all_v<
                cpp17::is_same_v<db::remove_all_prefixes<WrappedTags>,
                                 db::remove_all_prefixes<Tags>>...>> = nullptr>
  SPECTRE_ALWAYS_INLINE Variables& operator-=(
      const Variables<tmpl::list<WrappedTags...>>& rhs) noexcept {
    variable_data_ -= rhs.variable_data_;
    return *this;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE Variables& operator-=(
      const blaze::Vector<VT, VF>& rhs) noexcept {
    variable_data_ -= rhs;
    return *this;
  }

  SPECTRE_ALWAYS_INLINE Variables& operator*=(const value_type& rhs) noexcept {
    variable_data_ *= rhs;
    return *this;
  }

  SPECTRE_ALWAYS_INLINE Variables& operator/=(const value_type& rhs) noexcept {
    variable_data_ /= rhs;
    return *this;
  }

  template <typename... WrappedTags,
            Requires<tmpl2::flat_all_v<
                cpp17::is_same_v<db::remove_all_prefixes<WrappedTags>,
                                 db::remove_all_prefixes<Tags>>...>> = nullptr>
  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator+(
      const Variables<tmpl::list<WrappedTags...>>& lhs,
      const Variables& rhs) noexcept {
    return lhs.get_variable_data() + rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator+(
      const blaze::Vector<VT, VF>& lhs, const Variables& rhs) noexcept {
    return ~lhs + rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator+(
      const Variables& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.variable_data_ + ~rhs;
  }

  template <typename... WrappedTags,
            Requires<tmpl2::flat_all_v<
                cpp17::is_same_v<db::remove_all_prefixes<WrappedTags>,
                                 db::remove_all_prefixes<Tags>>...>> = nullptr>
  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator-(
      const Variables<tmpl::list<WrappedTags...>>& lhs,
      const Variables& rhs) noexcept {
    return lhs.get_variable_data() - rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator-(
      const blaze::Vector<VT, VF>& lhs, const Variables& rhs) noexcept {
    return ~lhs - rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator-(
      const Variables& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.variable_data_ - ~rhs;
  }

  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator*(
      const Variables& lhs, const value_type& rhs) noexcept {
    return lhs.variable_data_ * rhs;
  }
  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator*(
      const value_type& lhs, const Variables& rhs) noexcept {
    return lhs * rhs.variable_data_;
  }

  friend SPECTRE_ALWAYS_INLINE decltype(auto) operator/(
      const Variables& lhs, const value_type& rhs) noexcept {
    return lhs.variable_data_ / rhs;
  }

 private:
  //{@
  /*!
   * \brief Subscript operator
   *
   * The subscript operator is private since it should not be used directly.
   * Mathematical operations should be done using the math operators provided.
   * Since the internal ordering of variables is implementation defined there
   * is no safe way to perform any operation that is not a linear combination of
   * Variables. Retrieving a Tensor must be done via the `get()` function.
   *
   *  \requires `i >= 0 and i < size()`
   */
  SPECTRE_ALWAYS_INLINE value_type& operator[](const size_type i) noexcept {
    return variable_data_[i];
  }
  SPECTRE_ALWAYS_INLINE const value_type& operator[](const size_type i) const
      noexcept {
    return variable_data_[i];
  }
  //@}

  static SPECTRE_ALWAYS_INLINE void add_reference_variable_data(
      tmpl::list<> /*unused*/, const size_t /*variable_offset*/ = 0) noexcept {}

  template <
      typename TagToAdd, typename... Rest,
      Requires<tt::is_a<Tensor, typename TagToAdd::type>::value> = nullptr>
  void add_reference_variable_data(tmpl::list<TagToAdd, Rest...> /*unused*/,
                                   size_t variable_offset = 0) noexcept;

  friend bool operator==(const Variables& lhs, const Variables& rhs) noexcept {
    return lhs.variable_data_ == rhs.variable_data_;
  }

  template <class FriendTags>
  friend class Variables;

  std::unique_ptr<value_type[], decltype(&free)> variable_data_impl_{nullptr,
                                                                     &free};
  size_t size_ = 0;
  size_t number_of_grid_points_ = 0;

  // variable_data_ is only used to plug into the Blaze expression templates
  pointer_type variable_data_;
  tuples::TaggedTuple<Tags...> reference_variable_data_;
};

template <typename... Tags>
Variables<tmpl::list<Tags...>>::Variables() noexcept {
  // This makes an assertion trigger if one tries to assign to
  // components of a default-constructed Variables.
  const auto set_refs = [](auto& var) noexcept {
    for (auto& dv : var) {
      dv.set_data_ref(nullptr, 0);
    }
    return 0;
  };
  (void)set_refs;
  expand_pack(set_refs(tuples::get<Tags>(reference_variable_data_))...);
}

template <typename... Tags>
Variables<tmpl::list<Tags...>>::Variables(
    const size_t number_of_grid_points) noexcept {
  initialize(number_of_grid_points);
}

template <typename... Tags>
Variables<tmpl::list<Tags...>>::Variables(const size_t number_of_grid_points,
                                          const value_type value) noexcept {
  initialize(number_of_grid_points, value);
}

template <typename... Tags>
void Variables<tmpl::list<Tags...>>::initialize(
    const size_t number_of_grid_points) noexcept {
  size_ = number_of_grid_points * number_of_independent_components;
  if (size_ > 0) {
    // clang-tidy: cppcoreguidelines-no-malloc
    variable_data_impl_.reset(static_cast<value_type*>(
        malloc(number_of_grid_points *  // NOLINT
               number_of_independent_components * sizeof(value_type))));
    number_of_grid_points_ = number_of_grid_points;
#if defined(SPECTRE_DEBUG) || defined(SPECTRE_NAN_INIT)
    std::fill(variable_data_impl_.get(), variable_data_impl_.get() + size_,
              make_signaling_NaN<value_type>());
#endif  // SPECTRE_DEBUG
    variable_data_.reset(variable_data_impl_.get(), size_);
    add_reference_variable_data(tmpl::list<Tags...>{});
  }
}

template <typename... Tags>
void Variables<tmpl::list<Tags...>>::initialize(
    const size_t number_of_grid_points, const value_type value) noexcept {
  size_ = number_of_grid_points * number_of_independent_components;
  if (size_ > 0) {
    // clang-tidy: cppcoreguidelines-no-malloc
    variable_data_impl_.reset(static_cast<value_type*>(
        malloc(number_of_grid_points *  // NOLINT
               number_of_independent_components * sizeof(value_type))));
    number_of_grid_points_ = number_of_grid_points;
    std::fill(variable_data_impl_.get(), variable_data_impl_.get() + size_,
              value);
    variable_data_.reset(variable_data_impl_.get(), size_);
    add_reference_variable_data(tmpl::list<Tags...>{});
  }
}

/// \cond HIDDEN_SYMBOLS
template <typename... Tags>
Variables<tmpl::list<Tags...>>::Variables(
    const Variables<tmpl::list<Tags...>>& rhs) noexcept
    : size_(rhs.size_), number_of_grid_points_(rhs.number_of_grid_points()) {
  if (size_ > 0) {
    // clang-tidy: cppcoreguidelines-no-malloc
    variable_data_impl_.reset(static_cast<value_type*>(
        malloc(size_ * sizeof(value_type))));  // NOLINT
    variable_data_.reset(variable_data_impl_.get(), size_);
    add_reference_variable_data(tmpl::list<Tags...>{});
    variable_data_ =
        static_cast<const blaze::Vector<pointer_type, transpose_flag>&>(
            rhs.variable_data_);
  }
}

template <typename... Tags>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    const Variables<tmpl::list<Tags...>>& rhs) noexcept {
  if (&rhs == this) {
    return *this;
  }
  size_ = rhs.size_;
  if (number_of_grid_points_ != rhs.number_of_grid_points()) {
    number_of_grid_points_ = rhs.number_of_grid_points();
    if (size_ > 0) {
      // clang-tidy: cppcoreguidelines-no-malloc
      variable_data_impl_.reset(static_cast<value_type*>(
          malloc(size_ * sizeof(value_type))));  // NOLINT
      variable_data_.reset(variable_data_impl_.get(), size_);
      add_reference_variable_data(tmpl::list<Tags...>{});
    }
  }
  if (size_ > 0) {
    variable_data_ =
        static_cast<const blaze::Vector<pointer_type, transpose_flag>&>(
            rhs.variable_data_);
  }
  return *this;
}

template <typename... Tags>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    Variables<tmpl::list<Tags...>>&& rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }
  variable_data_impl_ = std::move(rhs.variable_data_impl_);
  size_ = rhs.size_;
  number_of_grid_points_ = std::move(rhs.number_of_grid_points_);
  variable_data_.reset(variable_data_impl_.get(), size());
  add_reference_variable_data(tmpl::list<Tags...>{});
  return *this;
}

template <typename... Tags>
template <typename... WrappedTags,
          Requires<tmpl2::flat_all<
              std::is_same<db::remove_all_prefixes<WrappedTags>,
                           db::remove_all_prefixes<Tags>>::value...>::value>>
Variables<tmpl::list<Tags...>>::Variables(
    const Variables<tmpl::list<WrappedTags...>>& rhs) noexcept
    : size_(rhs.size_), number_of_grid_points_(rhs.number_of_grid_points()) {
  if (size_ > 0) {
    // clang-tidy: cppcoreguidelines-no-malloc
    variable_data_impl_.reset(static_cast<value_type*>(
        malloc(size_ * sizeof(value_type))));  // NOLINT
    variable_data_.reset(variable_data_impl_.get(), size_);
    variable_data_ =
        static_cast<const blaze::Vector<pointer_type, transpose_flag>&>(
            rhs.variable_data_);
    add_reference_variable_data(tmpl::list<Tags...>{});
  }
}

template <typename... Tags>
template <typename... WrappedTags,
          Requires<tmpl2::flat_all_v<
              std::is_same<db::remove_all_prefixes<WrappedTags>,
                           db::remove_all_prefixes<Tags>>::value...>>>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    const Variables<tmpl::list<WrappedTags...>>& rhs) noexcept {
  size_ = rhs.size_;
  if (number_of_grid_points_ != rhs.number_of_grid_points()) {
    number_of_grid_points_ = rhs.number_of_grid_points();
    if (size_ > 0) {
      // clang-tidy: cppcoreguidelines-no-malloc
      variable_data_impl_.reset(static_cast<value_type*>(
          malloc(size_ * sizeof(value_type))));  // NOLINT
      variable_data_.reset(variable_data_impl_.get(), size_);
      add_reference_variable_data(tmpl::list<Tags...>{});
    }
  }
  variable_data_ =
      static_cast<const blaze::Vector<pointer_type, transpose_flag>&>(
          rhs.variable_data_);
  return *this;
}

template <typename... Tags>
template <typename... WrappedTags,
          Requires<tmpl2::flat_all<
              std::is_same<db::remove_all_prefixes<WrappedTags>,
                           db::remove_all_prefixes<Tags>>::value...>::value>>
Variables<tmpl::list<Tags...>>::Variables(
    Variables<tmpl::list<WrappedTags...>>&& rhs) noexcept
    : variable_data_impl_(std::move(rhs.variable_data_impl_)),
      size_(rhs.size()),
      number_of_grid_points_(rhs.number_of_grid_points()),
      variable_data_(variable_data_impl_.get(), size_),
      reference_variable_data_(std::move(rhs.reference_variable_data_)) {}

template <typename... Tags>
template <typename... WrappedTags,
          Requires<tmpl2::flat_all<
              std::is_same<db::remove_all_prefixes<WrappedTags>,
                           db::remove_all_prefixes<Tags>>::value...>::value>>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    Variables<tmpl::list<WrappedTags...>>&& rhs) noexcept {
  variable_data_impl_ = std::move(rhs.variable_data_impl_);
  size_ = rhs.size_;
  number_of_grid_points_ = std::move(rhs.number_of_grid_points_);
  variable_data_.reset(variable_data_impl_.get(), size());
  add_reference_variable_data(tmpl::list<Tags...>{});
  return *this;
}

template <typename... Tags>
void Variables<tmpl::list<Tags...>>::pup(PUP::er& p) noexcept {
  p | size_;
  p | number_of_grid_points_;
  if (p.isUnpacking()) {
    // clang-tidy: cppcoreguidelines-no-malloc
    variable_data_impl_.reset(static_cast<value_type*>(
        malloc(number_of_grid_points_ *  // NOLINT
               number_of_independent_components * sizeof(value_type))));
    variable_data_.reset(variable_data_impl_.get(), size());
    add_reference_variable_data(tmpl::list<Tags...>{});
  }
  PUParray(p, variable_data_impl_.get(), size_);
}
/// \endcond

// {@
/*!
 * \ingroup DataStructuresGroup
 * \brief Return Tag::type pointing into the contiguous array
 *
 * \tparam Tag the variable to return
 */
template <typename Tag, typename TagList>
constexpr typename Tag::type& get(Variables<TagList>& v) noexcept {
  static_assert(tmpl::list_contains_v<TagList, Tag>,
                "Could not retrieve Tag from Variables. See the first "
                "template parameter of the instantiation for what Tag is "
                "being retrieved and the second template parameter for "
                "what Tags are available.");
  return tuples::get<Tag>(v.reference_variable_data_);
}
template <typename Tag, typename TagList>
constexpr const typename Tag::type& get(const Variables<TagList>& v) noexcept {
  static_assert(tmpl::list_contains_v<TagList, Tag>,
                "Could not retrieve Tag from Variables. See the first "
                "template parameter of the instantiation for what Tag is "
                "being retrieved and the second template parameter for "
                "what Tags are available.");
  return tuples::get<Tag>(v.reference_variable_data_);
}
// @}

template <typename... Tags>
template <typename VT, bool VF>
Variables<tmpl::list<Tags...>>::Variables(
    const blaze::Vector<VT, VF>& expression) noexcept
    : size_((~expression).size()),
      number_of_grid_points_(size_ / number_of_independent_components) {
  initialize(number_of_grid_points_);
  variable_data_ = expression;
}

/// \cond
template <typename... Tags>
template <typename VT, bool VF>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    const blaze::Vector<VT, VF>& expression) noexcept {
  if (size_ != (~expression).size()) {
    size_ = (~expression).size();
    number_of_grid_points_ = size_ / number_of_independent_components;
    initialize(number_of_grid_points_);
  }
  variable_data_ = expression;
  return *this;
}
/// \endcond

/// \cond HIDDEN_SYMBOLS
template <typename... Tags>
template <typename TagToAdd, typename... Rest,
          Requires<tt::is_a<Tensor, typename TagToAdd::type>::value>>
void Variables<tmpl::list<Tags...>>::add_reference_variable_data(
    tmpl::list<TagToAdd, Rest...> /*unused*/,
    const size_t variable_offset) noexcept {
  ASSERT(size_ > (variable_offset + TagToAdd::type::size() - 1) *
                     number_of_grid_points_,
         "This ASSERT is typically triggered because a Variables class was "
         "default constructed. The only reason the Variables class has a "
         "default constructor is because Charm++ uses it, you are not "
         "supposed to use it otherwise.");
  typename TagToAdd::type& var =
      tuples::get<TagToAdd>(reference_variable_data_);
  for (size_t i = 0; i < TagToAdd::type::size(); ++i) {
    var[i].set_data_ref(
        &variable_data_[(variable_offset + i) * number_of_grid_points_],
        number_of_grid_points_);
  }
  add_reference_variable_data(tmpl::list<Rest...>{},
                              variable_offset + TagToAdd::type::size());
}
/// \endcond

template <typename... Tags>
Variables<tmpl::list<Tags...>>& operator*=(
    Variables<tmpl::list<Tags...>>& lhs,
    const typename Variables<tmpl::list<Tags...>>::vector_type& rhs) noexcept {
  using value_type = typename Variables<tmpl::list<Tags...>>::value_type;
  ASSERT(lhs.number_of_grid_points() == rhs.size(),
         "Size mismatch in multiplication: " << lhs.number_of_grid_points()
                                             << " and " << rhs.size());
  value_type* const lhs_data = lhs.data();
  const value_type* const rhs_data = rhs.data();
  for (size_t c = 0; c < lhs.number_of_independent_components; ++c) {
    for (size_t s = 0; s < lhs.number_of_grid_points(); ++s) {
      // clang-tidy: do not use pointer arithmetic
      lhs_data[c * lhs.number_of_grid_points() + s] *= rhs_data[s];  // NOLINT
    }
  }
  return lhs;
}

template <typename... Tags>
Variables<tmpl::list<Tags...>> operator*(
    const Variables<tmpl::list<Tags...>>& lhs,
    const typename Variables<tmpl::list<Tags...>>::vector_type& rhs) noexcept {
  auto result = lhs;
  result *= rhs;
  return result;
}

template <typename... Tags>
Variables<tmpl::list<Tags...>> operator*(
    const typename Variables<tmpl::list<Tags...>>::vector_type& lhs,
    const Variables<tmpl::list<Tags...>>& rhs) noexcept {
  auto result = rhs;
  result *= lhs;
  return result;
}

template <typename... Tags>
Variables<tmpl::list<Tags...>>& operator/=(
    Variables<tmpl::list<Tags...>>& lhs,
    const typename Variables<tmpl::list<Tags...>>::vector_type& rhs) noexcept {
  ASSERT(lhs.number_of_grid_points() == rhs.size(),
         "Size mismatch in multiplication: " << lhs.number_of_grid_points()
                                             << " and " << rhs.size());
  using value_type = typename Variables<tmpl::list<Tags...>>::value_type;
  value_type* const lhs_data = lhs.data();
  const value_type* const rhs_data = rhs.data();
  for (size_t c = 0; c < lhs.number_of_independent_components; ++c) {
    for (size_t s = 0; s < lhs.number_of_grid_points(); ++s) {
      // clang-tidy: do not use pointer arithmetic
      lhs_data[c * lhs.number_of_grid_points() + s] /= rhs_data[s];  // NOLINT
    }
  }
  return lhs;
}

template <typename... Tags>
Variables<tmpl::list<Tags...>> operator/(
    const Variables<tmpl::list<Tags...>>& lhs,
    const typename Variables<tmpl::list<Tags...>>::vector_type& rhs) noexcept {
  auto result = lhs;
  result /= rhs;
  return result;
}

namespace Variables_detail {
template <typename TagsList>
std::ostream& print_helper(std::ostream& os, const Variables<TagsList>& /*d*/,
                           tmpl::list<> /*meta*/) noexcept {
  return os << "Variables is empty!";
}

template <typename Tag, typename TagsList>
std::ostream& print_helper(std::ostream& os, const Variables<TagsList>& d,
                           tmpl::list<Tag> /*meta*/) noexcept {
  return os << pretty_type::short_name<Tag>() << ":\n" << get<Tag>(d);
}

template <typename Tag, typename SecondTag, typename... RemainingTags,
          typename TagsList>
std::ostream& print_helper(
    std::ostream& os, const Variables<TagsList>& d,
    tmpl::list<Tag, SecondTag, RemainingTags...> /*meta*/) noexcept {
  os << pretty_type::short_name<Tag>() << ":\n";
  os << get<Tag>(d) << "\n\n";
  print_helper(os, d, tmpl::list<SecondTag, RemainingTags...>{});
  return os;
}
}  // namespace Variables_detail

template <typename TagsList>
std::ostream& operator<<(std::ostream& os,
                         const Variables<TagsList>& d) noexcept {
  return Variables_detail::print_helper(os, d, TagsList{});
}

template <typename TagsList>
bool operator!=(const Variables<TagsList>& lhs,
                const Variables<TagsList>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \ingroup DataStructuresGroup
/// Construct a variables from the `Tensor`s in a `TaggedTuple`.
template <typename... Tags>
Variables<tmpl::list<Tags...>> variables_from_tagged_tuple(
    const tuples::TaggedTuple<Tags...>& tuple) noexcept {
  auto result = make_with_value<Variables<tmpl::list<Tags...>>>(
      get<tmpl::front<tmpl::list<Tags...>>>(tuple), 0.0);
  result.assign_subset(tuple);
  return result;
}

namespace MakeWithValueImpls {
template <typename TagList>
struct MakeWithValueImpl<Variables<TagList>,
                         typename Variables<TagList>::vector_type> {
  /// \brief Returns a Variables whose vectors are the same size as `input`,
  /// with each element equal to `value`.
  static SPECTRE_ALWAYS_INLINE Variables<TagList> apply(
      const typename Variables<TagList>::vector_type& input,
      const typename Variables<TagList>::value_type value) noexcept {
    return Variables<TagList>(input.size(), value);
  }
};

template <typename TagList, typename... Structure>
struct MakeWithValueImpl<
    Variables<TagList>,
    Tensor<typename Variables<TagList>::vector_type, Structure...>> {
  /// \brief Returns a Variables whose DataVectors are the same size as `input`,
  /// with each element equal to `value`.
  static SPECTRE_ALWAYS_INLINE Variables<TagList> apply(
      const Tensor<typename Variables<TagList>::vector_type, Structure...>&
          input,
      const typename Variables<TagList>::value_type value) noexcept {
    return Variables<TagList>(input.begin()->size(), value);
  }
};

template <typename TagListOut, typename TagListIn>
struct MakeWithValueImpl<Variables<TagListOut>, Variables<TagListIn>> {
  /// \brief Returns a Variables whose DataVectors are the same size as `input`,
  /// with each element equal to `value`.
  static SPECTRE_ALWAYS_INLINE Variables<TagListOut> apply(
      const Variables<TagListIn>& input,
      const typename Variables<TagListIn>::value_type value) noexcept {
    return Variables<TagListOut>(input.number_of_grid_points(), value);
  }
};
}  // namespace MakeWithValueImpls

namespace db {
template <typename TagList, typename Tag>
struct Subitems<TagList, Tag,
                Requires<tt::is_a_v<Variables, item_type<Tag, TagList>>>> {
  using type = typename item_type<Tag>::tags_list;

  template <typename Subtag>
  static void create_item(
      const gsl::not_null<item_type<Tag>*> parent_value,
      const gsl::not_null<item_type<Subtag>*> sub_value) noexcept {
    auto& vars = get<Subtag>(*parent_value);
    // Only update the Tensor if the Variables has changed its allocation
    if (vars.begin()->data() != sub_value->begin()->data()) {
      for (auto vars_it = vars.begin(), sub_var_it = sub_value->begin();
           vars_it != vars.end(); ++vars_it, ++sub_var_it) {
        sub_var_it->set_data_ref(make_not_null(&*vars_it));
      }
    }
  }

  template <typename Subtag>
  static const item_type<Subtag>& create_compute_item(
      const item_type<Tag>& parent_value) noexcept {
    return get<Subtag>(parent_value);
  }
};
}  // namespace db

namespace Tags {
template <size_t N, typename T>
struct TempTensor {
  using type = T;
  static std::string name() noexcept {
    return std::string("TempTensor") + std::to_string(N);
  }
};

// @{
/// \ingroup PeoGroup
/// Variables Tags for temporary tensors inside a function.
template <size_t N, typename DataType = DataVector>
using TempScalar = TempTensor<N, Scalar<DataType>>;

// Rank 1
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempa = TempTensor<N, tnsr::a<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempA = TempTensor<N, tnsr::A<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempi = TempTensor<N, tnsr::i<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempI = TempTensor<N, tnsr::I<DataType, SpatialDim, Fr>>;

// Rank 2
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempab = TempTensor<N, tnsr::ab<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempaB = TempTensor<N, tnsr::aB<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempAb = TempTensor<N, tnsr::Ab<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempAB = TempTensor<N, tnsr::AB<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempij = TempTensor<N, tnsr::ij<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempiJ = TempTensor<N, tnsr::iJ<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempIj = TempTensor<N, tnsr::Ij<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempIJ = TempTensor<N, tnsr::IJ<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempia = TempTensor<N, tnsr::ia<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempaa = TempTensor<N, tnsr::aa<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempAA = TempTensor<N, tnsr::AA<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempii = TempTensor<N, tnsr::ii<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempII = TempTensor<N, tnsr::II<DataType, SpatialDim, Fr>>;

// Rank 3
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempijj = TempTensor<N, tnsr::ijj<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempiaa = TempTensor<N, tnsr::iaa<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempabb = TempTensor<N, tnsr::abb<DataType, SpatialDim, Fr>>;
// @}
}  // namespace Tags
