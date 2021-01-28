// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>  // IWYU pragma: keep  // for std::move

#include "DataStructures/TempBuffer.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/*!
 * \ingroup DataStructuresGroup
 * A temporary buffer with contents computed on demand.
 *
 * When an entry in the buffer is first requested, it is computed by
 * the `Computer` class.  Subsequent requests just return the cached
 * value.  The computer can itself request data from the cache to use
 * in its computations.
 *
 * For the cache
 * \snippet Test_CachedTempBuffer.cpp alias
 * the function used to compute `Tags::Scalar2<DataType>` is
 * \snippet Test_CachedTempBuffer.cpp compute_func
 */
template <typename Computer, typename... Tags>
class CachedTempBuffer {
 public:
  /// Construct the buffer with the given computer.  `size` is passed
  /// to the underlying `TempBuffer` constructor.
  CachedTempBuffer(const size_t size, Computer computer) noexcept
      : data_(size), computer_(std::move(computer)) {}

  /// Obtain a value from the buffer, computing it if necessary.
  template <typename Tag>
  const typename Tag::type& get_var(Tag /*meta*/) noexcept {
    static_assert(tmpl::list_contains_v<tmpl::list<Tags...>, Tag>,
                  "The requested tag is not available. See the template "
                  "parameters of 'CachedTempBuffer' for the computer type and "
                  "the available tags, and the template parameter of the "
                  "'get_var' function for the requested tag.");
    // This function can't be called "get" because that interferes
    // with the ADL needed to access data_.
    if (not get<Computed<Tag>>(computed_flags_)) {
      computer_(make_not_null(&get<Tag>(data_)), make_not_null(this), Tag{});
      get<Computed<Tag>>(computed_flags_) = true;
    }
    return get<Tag>(data_);
  }

 private:
  template <typename Tag>
  struct Computed {
    using type = bool;
  };

  TempBuffer<tmpl::list<Tags...>> data_;
  tuples::TaggedTuple<Computed<Tags>...> computed_flags_{
      ((void)Tags{}, false)...};
  Computer computer_;
};
