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
 * When an entry in the buffer is first requested, it is computed by the
 * `computer` that is passed to the `get_var` function. Subsequent requests just
 * return the cached value.  The computer can itself request data from the cache
 * to use in its computations.
 *
 * For the cache
 * \snippet Test_CachedTempBuffer.cpp alias
 * the function used to compute `Tags::Scalar2<DataType>` is
 * \snippet Test_CachedTempBuffer.cpp compute_func
 */
template <typename... Tags>
class CachedTempBuffer {
 public:
  using tags_list = tmpl::list<Tags...>;

  /// Construct the buffer with the given computer.  `size` is passed
  /// to the underlying `TempBuffer` constructor.
  CachedTempBuffer(const size_t size) : data_(size) {}

  /// Obtain a value from the buffer, computing it if necessary.
  template <typename Computer, typename Tag>
  const typename Tag::type& get_var(const Computer& computer, Tag /*meta*/) {
    static_assert(tmpl::list_contains_v<tmpl::list<Tags...>, Tag>,
                  "The requested tag is not available. See the template "
                  "parameters of 'CachedTempBuffer' for the computer type and "
                  "the available tags, and the template parameter of the "
                  "'get_var' function for the requested tag.");
    // This function can't be called "get" because that interferes
    // with the ADL needed to access data_.
    if (not get<Computed<Tag>>(computed_flags_)) {
      computer(make_not_null(&get<Tag>(data_)), make_not_null(this), Tag{});
      get<Computed<Tag>>(computed_flags_) = true;
    }
    return get<Tag>(data_);
  }
  size_t number_of_grid_points() const { return data_.number_of_grid_points(); }

 private:
  template <typename Tag>
  struct Computed {
    using type = bool;
  };

  TempBuffer<tmpl::list<Tags...>> data_;
  tuples::TaggedTuple<Computed<Tags>...> computed_flags_{
      ((void)Tags{}, false)...};
};

/// Instantiate a `CachedTempBuffer` from a typelist instead of a parameter pack
template <typename Tags>
using cached_temp_buffer_from_typelist = tmpl::wrap<Tags, CachedTempBuffer>;
