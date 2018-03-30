// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functions for serializing factory-created classes

#pragma once

#include <typeinfo>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
namespace register_derived_classes_with_charm_detail {
template <typename Type>
SPECTRE_ALWAYS_INLINE void make_type_pupable() noexcept {
  // This assumes typeid(Type).name() is different for each registered
  // type.  If we don't trust that we can add a counter or something.
  PUPable_reg2(Type, typeid(Type).name())
}

template <typename... Types>
SPECTRE_ALWAYS_INLINE void make_list_pupable(
    const tmpl::list<Types...> /*meta*/) noexcept {
  expand_pack((make_type_pupable<Types>(), 0)...);
}
}  // namespace register_derived_classes_with_charm_detail

template <typename Base>
SPECTRE_ALWAYS_INLINE void register_derived_classes_with_charm() noexcept {
  register_derived_classes_with_charm_detail::make_list_pupable(
      typename Base::creatable_classes{});
}
}  // namespace Parallel
