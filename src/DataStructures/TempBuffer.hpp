// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/*!
 * \ingroup DataStructuresGroup
 * \brief A TempBuffer holds a set of `Tensor<DataType>`s, where
 * DataType is either a `DataVector` (or similar type) or a
 * fundamental type, in a way that minimizes allocations.
 *
 * The user gets references to Tensors inside of the TempBuffer using,
 * e.g., `auto& variable = get<Tag>(temp_buffer)`, where `Tag` is one
 * of the Tags in the `TagList`.
 *
 * If DataType is a DataVector or similar, than TempBuffer is a
 * Variables.  If DataType is a fundamental type, then TempBuffer is a
 * TaggedTuple.
 *
 */
template <typename TagList,
          bool is_fundamental = cpp17::is_fundamental_v<
              typename tmpl::front<TagList>::type::value_type>>
struct TempBuffer;

template <typename TagList>
struct TempBuffer<TagList, true> : tuples::tagged_tuple_from_typelist<TagList> {
  explicit TempBuffer(const size_t /*size*/) noexcept
      : tuples::tagged_tuple_from_typelist<TagList>::TaggedTuple(){}
};

template <typename TagList>
struct TempBuffer<TagList, false> : Variables<TagList> {
  using Variables<TagList>::Variables;
};
