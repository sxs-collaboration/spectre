// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::util {
/// Get data for the tensors in `TagsList` from the `analytic_data`. The
/// `analytic_data` can be any subclass of `Base` that is listed in
/// `Metavariables::factory_creation`.
template <typename TagsList, typename Base, typename DbTagsList,
          typename... Args>
Variables<TagsList> get_analytic_data(const Base& analytic_data,
                                      const db::DataBox<DbTagsList>& box,
                                      const Args&... args) {
  using factory_classes =
      typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
          box))>::factory_creation::factory_classes;
  return call_with_dynamic_type<Variables<TagsList>,
                                tmpl::at<factory_classes, Base>>(
      &analytic_data, [&args...](const auto* const derived) {
        return variables_from_tagged_tuple(
            derived->variables(args..., TagsList{}));
      });
}
}  // namespace elliptic::util
