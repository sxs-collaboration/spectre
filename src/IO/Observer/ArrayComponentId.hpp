// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <ckarrayindex.h>
#include <cstddef>
#include <functional>
#include <string>

#include "Utilities/PrettyType.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief An ID type that identifies both the parallel component and the index
 * in the parallel component.
 *
 * A specialization of `std::hash` is provided to allow using `ArrayComponentId`
 * as a key in associative containers.
 */
class ArrayComponentId {
 public:
  ArrayComponentId() = default;  // Needed for Charm++ serialization

  template <typename ParallelComponent>
  ArrayComponentId(const ParallelComponent* /*meta*/,
                   const CkArrayIndex& index) noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  size_t component_id() const noexcept { return component_id_; }

  const CkArrayIndex& array_index() const noexcept { return array_index_; }

 private:
  size_t component_id_{0};
  CkArrayIndex array_index_{};
};

template <typename ParallelComponent>
ArrayComponentId::ArrayComponentId(const ParallelComponent* const /*meta*/,
                                   const CkArrayIndex& index) noexcept
    : component_id_(
          std::hash<std::string>{}(pretty_type::get_name<ParallelComponent>())),
      array_index_(index) {}

bool operator==(const ArrayComponentId& lhs,
                const ArrayComponentId& rhs) noexcept;

bool operator!=(const ArrayComponentId& lhs,
                const ArrayComponentId& rhs) noexcept;
}  // namespace observers

namespace std {
template <>
struct hash<observers::ArrayComponentId> {
  size_t operator()(const observers::ArrayComponentId& t) const {
    size_t result = t.component_id();
    boost::hash_combine(result, t.array_index().hash());
    return result;
  }
};
}  // namespace std
