// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Variables

#pragma once

#include "DataStructures/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <typename X, typename Symm, typename IndexLs>
class Tensor;

template <typename TagsLs>
class Variables;
/// \endcond

namespace Tags {
template <typename TagsLs>
struct Variables : db::DataBoxTag {
  static_assert(tt::is_a<tmpl::list, TagsLs>::value,
                "The TagsLs passed to Tags::Variables is not a typelist");
  using tags_list = TagsLs;
  using type = ::Variables<TagsLs>;
  static constexpr db::DataBoxString_t label = "Variables";
};
}  // namespace Tags

/// \cond
template <typename TagsLs>
class Variables;
/// \endcond

/*!
 * \ingroup DataStructures
 * \brief A Variables holds a contiguous memory block with Tensors pointing
 * into it
 */
template <typename... Tags>
class Variables<tmpl::list<Tags...>> {
 public:
  using value_type = double;
  using allocator_type = std::allocator<value_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  /// A typelist of the Tags whose variables are held
  using tags_list = tmpl::list<Tags...>;

  /// The number of variables of the Variables object is holding. E.g.
  /// \f$\psi_{ab}\f$ would be counted as one variable.
  static constexpr auto number_of_variables = sizeof...(Tags);

  /// \cond
  // If you encounter an error of the `size()` function not existing you are
  // not filling the Variables with Tensors. Variables can be generalized to
  // holding containers other than Tensor by having the contaiers have a
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
  Variables() = default;

  explicit Variables(
      size_t number_of_grid_points,
      double value = std::numeric_limits<double>::signaling_NaN());

  Variables(Variables&& rhs) noexcept = default;
  Variables& operator=(Variables&& rhs) noexcept;

  Variables(const Variables& rhs);
  Variables& operator=(const Variables& rhs);

  /// \cond HIDDEN_SYMBOLS
  ~Variables() noexcept = default;
  /// \endcond

  constexpr size_t number_of_grid_points() const noexcept {
    return number_of_grid_points_;
  }

  /// Number of grid points * number of independent components
  constexpr size_type size() const noexcept { return size_; }

  //{@
  /// Access pointer to underlying data
  double* data() noexcept { return variable_data_.data(); }
  const double* data() const noexcept { return variable_data_.data(); }
  //@}

  // {@
  /*!
   *  \brief Return Tag::type pointing into the contiguous array
   *
   *  \tparam Tag the variable to return
   */
  template <typename Tag>
  constexpr auto& get() noexcept {
    return tuples::get<Tag>(reference_variable_data_);
  }
  template <typename Tag>
  constexpr const auto& get() const noexcept {
    return tuples::get<Tag>(reference_variable_data_);
  }
  // @}

  /// Serialization for Charm++.
  void pup(PUP::er& p);

  /// Converting constructor for an expression to a Variables class
  // clang-tidy: mark as explicit (we want conversion to Variables)
  template <typename VT, bool VF>
  Variables(const blaze::Vector<VT, VF>& expression);  // NOLINT

  template <typename VT, bool VF>
  Variables& operator=(const blaze::Vector<VT, VF>& expression);

  Variables& operator+=(const Variables& rhs) {
    variable_data_ += rhs.variable_data_;
    return *this;
  }
  template <typename VT, bool VF>
  Variables& operator+=(const blaze::Vector<VT, VF>& rhs) {
    variable_data_ += rhs;
    return *this;
  }

  Variables& operator-=(const Variables& rhs) {
    variable_data_ -= rhs.variable_data_;
    return *this;
  }
  template <typename VT, bool VF>
  Variables& operator-=(const blaze::Vector<VT, VF>& rhs) {
    variable_data_ -= rhs;
    return *this;
  }

  Variables& operator*=(const double& rhs) {
    variable_data_ *= rhs;
    return *this;
  }

  Variables& operator/=(const double& rhs) {
    variable_data_ /= rhs;
    return *this;
  }

  friend decltype(auto) operator+(const Variables& lhs, const Variables& rhs) {
    return lhs.variable_data_ + rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend decltype(auto) operator+(const blaze::Vector<VT, VF>& lhs,
                                  const Variables& rhs) {
    return ~lhs + rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend decltype(auto) operator+(const Variables& lhs,
                                  const blaze::Vector<VT, VF>& rhs) {
    return lhs.variable_data_ + ~rhs;
  }

  friend decltype(auto) operator-(const Variables& lhs, const Variables& rhs) {
    return lhs.variable_data_ - rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend decltype(auto) operator-(const blaze::Vector<VT, VF>& lhs,
                                  const Variables& rhs) {
    return ~lhs - rhs.variable_data_;
  }
  template <typename VT, bool VF>
  friend decltype(auto) operator-(const Variables& lhs,
                                  const blaze::Vector<VT, VF>& rhs) {
    return lhs.variable_data_ - ~rhs;
  }

  friend decltype(auto) operator*(const Variables& lhs, const double& rhs) {
    return lhs.variable_data_ * rhs;
  }
  friend decltype(auto) operator*(const double& lhs, const Variables& rhs) {
    return lhs * rhs.variable_data_;
  }

  friend decltype(auto) operator/(const Variables& lhs, const double& rhs) {
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
  double& operator[](const size_type i) noexcept { return variable_data_[i]; }
  const double& operator[](const size_type i) const noexcept {
    return variable_data_[i];
  }
  //@}

  static SPECTRE_ALWAYS_INLINE void constexpr add_reference_variable_data(
      typelist<> /*unused*/, const size_t /*variable_offset*/ = 0) {
    CASSERT(sizeof...(Tags) > 0,
            "This ASSERT is triggered if you try to construct a Variables "
            "with no Tags. A Variables with no Tags is a valid type, but "
            "cannot be used in a meaningful way.");
  }

  template <
      typename TagToAdd, typename... Rest,
      Requires<tt::is_a<Tensor, typename TagToAdd::type>::value> = nullptr>
  void add_reference_variable_data(typelist<TagToAdd, Rest...> /*unused*/,
                                   size_t variable_offset = 0);

  friend bool operator==(const Variables& lhs, const Variables& rhs) noexcept {
    return lhs.variable_data_ == rhs.variable_data_;
  }

  std::vector<double, allocator_type> variable_data_impl_;
  // variable_data_ is only used to plug into the Blaze expression templates
  PointerVector<double> variable_data_;
  size_t size_ = 0;
  size_t number_of_grid_points_ = 0;
  tuples::TaggedTuple<Tags...> reference_variable_data_;
};

template <typename... Tags>
Variables<tmpl::list<Tags...>>::Variables(const size_t number_of_grid_points,
                                          const double value)
    : variable_data_impl_(
          number_of_grid_points * number_of_independent_components, value),
      variable_data_(variable_data_impl_.data(), variable_data_impl_.size()),
      size_(number_of_grid_points * number_of_independent_components),
      number_of_grid_points_(number_of_grid_points) {
  add_reference_variable_data(typelist<Tags...>{});
}

/// \cond HIDDEN_SYMBOLS
template <typename... Tags>
Variables<tmpl::list<Tags...>>::Variables(
    const Variables<tmpl::list<Tags...>>& rhs)
    : variable_data_impl_(rhs.variable_data_impl_),
      variable_data_(variable_data_impl_.data(), variable_data_impl_.size()),
      size_(rhs.size_),
      number_of_grid_points_(rhs.number_of_grid_points()) {
  add_reference_variable_data(typelist<Tags...>{});
}

template <typename... Tags>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    const Variables<tmpl::list<Tags...>>& rhs) {
  if (&rhs == this) {
    return *this;
  }
  variable_data_impl_ = rhs.variable_data_impl_;
  variable_data_.reset(variable_data_impl_.data(), variable_data_impl_.size());
  size_ = rhs.size_;
  number_of_grid_points_ = rhs.number_of_grid_points();
  add_reference_variable_data(typelist<Tags...>{});
  return *this;
}

template <typename... Tags>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    Variables<tmpl::list<Tags...>>&& rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }
  variable_data_impl_ = std::move(rhs.variable_data_impl_);
  variable_data_.reset(variable_data_impl_.data(), variable_data_impl_.size());
  size_ = rhs.size_;
  number_of_grid_points_ = std::move(rhs.number_of_grid_points_);
  add_reference_variable_data(typelist<Tags...>{});
  return *this;
}

template <typename... Tags>
void Variables<tmpl::list<Tags...>>::pup(PUP::er& p) {
  p | variable_data_impl_;
  p | size_;
  p | number_of_grid_points_;
  if (p.isUnpacking()) {
    variable_data_.reset(variable_data_impl_.data(),
                         variable_data_impl_.size());
    add_reference_variable_data(typelist<Tags...>{});
  }
}
/// \endcond

template <typename... Tags>
template <typename VT, bool VF>
Variables<tmpl::list<Tags...>>::Variables(
    const blaze::Vector<VT, VF>& expression)
    : variable_data_impl_((~expression).size()),
      variable_data_(variable_data_impl_.data(), variable_data_impl_.size()),
      size_((~expression).size()),
      number_of_grid_points_(size_ / number_of_independent_components) {
  variable_data_ = expression;
  add_reference_variable_data(typelist<Tags...>{});
}

/// \cond
template <typename... Tags>
template <typename VT, bool VF>
Variables<tmpl::list<Tags...>>& Variables<tmpl::list<Tags...>>::operator=(
    const blaze::Vector<VT, VF>& expression) {
  size_ = (~expression).size();
  number_of_grid_points_ = size_ / number_of_independent_components;
  variable_data_impl_.resize(size_);
  variable_data_.reset(variable_data_impl_.data(), variable_data_impl_.size());
  variable_data_ = expression;
  add_reference_variable_data(typelist<Tags...>{});
  return *this;
}
/// \endcond

/// \cond HIDDEN_SYMBOLS
template <typename... Tags>
template <typename TagToAdd, typename... Rest,
          Requires<tt::is_a<Tensor, typename TagToAdd::type>::value>>
void Variables<tmpl::list<Tags...>>::add_reference_variable_data(
    typelist<TagToAdd, Rest...> /*unused*/, const size_t variable_offset) {
  CASSERT(size_ > (variable_offset + TagToAdd::type::size() - 1) *
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
  add_reference_variable_data(typelist<Rest...>{},
                              variable_offset + TagToAdd::type::size());
}
/// \endcond

namespace Variables_detail {
template <typename TagsLs>
std::ostream& print_helper(std::ostream& os, const Variables<TagsLs>& /*d*/,
                           typelist<> /*meta*/) {
  return os << "Variables is empty!";
}

template <typename Tag, typename TagsLs>
std::ostream& print_helper(std::ostream& os, const Variables<TagsLs>& d,
                           typelist<Tag> /*meta*/) {
  return os << d.template get<Tag>();
}
template <typename Tag, typename SecondTag, typename... RemainingTags,
          typename TagsLs>
std::ostream& print_helper(
    std::ostream& os, const Variables<TagsLs>& d,
    typelist<Tag, SecondTag, RemainingTags...> /*meta*/) {
  os << d.template get<Tag>() << '\n';
  print_helper(os, d, typelist<SecondTag, RemainingTags...>{});
  return os;
}
}  // namespace Variables_detail

template <typename TagsLs>
std::ostream& operator<<(std::ostream& os, const Variables<TagsLs>& d) {
  return Variables_detail::print_helper(os, d, TagsLs{});
}

template <typename TagsLs>
bool operator!=(const Variables<TagsLs>& lhs, const Variables<TagsLs>& rhs) {
  return not(lhs == rhs);
}
