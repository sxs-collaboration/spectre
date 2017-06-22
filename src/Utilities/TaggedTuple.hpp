// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TaggedTuple

#pragma once

#include <tuple>
#include <type_traits>

#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup Utilities DataStructures
 * \brief An associative container that is indexed by structs
 *
 * A data structure that is indexed by Tags. A Tag is a struct that contains
 * a type alias named `type`, which is the type of the object stored with
 * index Tag.
 *
 * \tparam Tags the tags of the objects to be placed in the tuple
 */
template <typename... Tags>
struct TaggedTuple {
  static_assert(tmpl::is_set<Tags...>::value,
                "Cannot create a TaggedTuple with duplicate Tags.");
  /*!
   * \brief Construct a TaggedTuple with Args
   * \requires `std::is_convertible_v<Args, typename Tags::type>...` is `true`
   *
   * \example
   * \snippet Test_TaggedTuple.cpp construction_example
   */
  template <typename... Args>
  explicit TaggedTuple(Args&&... args) : data_(std::forward<Args>(args)...) {}

  /// Used for serialization with Charm++
  void pup(PUP::er& p) { p | data_; }  // NOLINT

  /// Returns the size of a TaggedTuple
  static constexpr std::size_t size() { return sizeof...(Tags); }

  // @{
  /*!
   * \brief Retrieve a stored object with tag `Tag`
   *
   * \example
   * \snippet Test_TaggedTuple.cpp get_example
   *
   * \tparam Tag the tag of the object to retrieve
   * \return reference to the object held in the tuple
   */
  template <typename Tag>
  constexpr typename Tag::type& get() noexcept {
    static_assert(
        not std::is_same<tmpl::index_of<tmpl::list<Tags...>, Tag>,
                         tmpl::no_such_type_>::value,
        "Could not retrieve Tag from TaggedTuple. See the first template "
        "parameter of the instantiation for what Tag is being "
        "retrieved and the remaining template parameters for what "
        "Tags are available.");
    return std::get<tmpl::index_of<tmpl::list<Tags...>, Tag>::value>(data_);
  }

  template <typename Tag>
  constexpr const typename Tag::type& get() const noexcept {
    static_assert(
        not std::is_same<tmpl::index_of<tmpl::list<Tags...>, Tag>,
                         tmpl::no_such_type_>::value,
        "Could not retrieve Tag from TaggedTuple. See the first template "
        "parameter of the instantiation for what Tag is being "
        "retrieved and the remaining template parameters for what "
        "Tags are available.");
    return std::get<tmpl::index_of<tmpl::list<Tags...>, Tag>::value>(data_);
  }
  // @}

  /// Stream operator for TaggedTuple
  friend std::ostream& operator<<(std::ostream& os,
                                  const TaggedTuple<Tags...>& t) {
    return os << t.data_;
  }

  constexpr friend bool operator==(const TaggedTuple& lhs,
                                   const TaggedTuple& rhs) {
    return lhs.data_ == rhs.data_;
  }
  constexpr friend bool operator!=(const TaggedTuple& lhs,
                                   const TaggedTuple& rhs) {
    return lhs.data_ != rhs.data_;
  }
  constexpr friend bool operator>(const TaggedTuple& lhs,
                                  const TaggedTuple& rhs) {
    return lhs.data_ > rhs.data_;
  }
  constexpr friend bool operator>=(const TaggedTuple& lhs,
                                   const TaggedTuple& rhs) {
    return lhs.data_ >= rhs.data_;
  }
  constexpr friend bool operator<(const TaggedTuple& lhs,
                                  const TaggedTuple& rhs) {
    return lhs.data_ < rhs.data_;
  }
  constexpr friend bool operator<=(const TaggedTuple& lhs,
                                   const TaggedTuple& rhs) {
    return lhs.data_ <= rhs.data_;
  }

 private:
  std::tuple<typename Tags::type...> data_;
};

namespace TaggedTuple_detail {
template <typename T>
struct TaggedTupleTypelistImpl;

template <template <typename...> class Ls, typename... Tags>
struct TaggedTupleTypelistImpl<Ls<Tags...>> {
  using type = TaggedTuple<Tags...>;
};
}  // namespace TaggedTuple_detail

/// \see TaggedTuple
template <typename Ls>
using TaggedTupleTypelist =
    typename TaggedTuple_detail::TaggedTupleTypelistImpl<Ls>::type;
