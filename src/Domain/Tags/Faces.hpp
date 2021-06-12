// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace domain {
namespace Tags {

/// The `Tag` on element faces
template <size_t Dim, typename Tag>
struct Faces : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = DirectionMap<Dim, typename Tag::type>;
};

}  // namespace Tags

/// Wrap `Tag` in `domain::Tags::Faces`, unless `Tag` is in the `VolumeTags`
/// list
template <typename Dim, typename Tag, typename VolumeTags = tmpl::list<>>
struct make_faces_tag {
  using type = tmpl::conditional_t<tmpl::list_contains_v<VolumeTags, Tag>, Tag,
                                   domain::Tags::Faces<Dim::value, Tag>>;
};

/// Wrap all tags in `TagsList` in `domain::Tags::Faces`, except those in the
/// `VolumeTags` list
template <size_t Dim, typename TagsList, typename VolumeTags = tmpl::list<>>
using make_faces_tags =
    tmpl::transform<TagsList, make_faces_tag<tmpl::pin<tmpl::size_t<Dim>>,
                                             tmpl::_1, tmpl::pin<VolumeTags>>>;

}  // namespace domain

namespace db {
template <size_t Dim, typename VariablesTag>
struct Subitems<domain::Tags::Faces<Dim, VariablesTag>,
                Requires<tt::is_a_v<Variables, typename VariablesTag::type>>> {
  template <typename LocalTag>
  using faces_tag = domain::Tags::Faces<Dim, LocalTag>;

  using tag = faces_tag<VariablesTag>;
  using type =
      db::wrap_tags_in<faces_tag, typename VariablesTag::type::tags_list>;

  template <typename Subtag>
  static void create_item(
      const gsl::not_null<typename tag::type*> parent_value,
      const gsl::not_null<typename Subtag::type*> sub_value) noexcept {
    sub_value->clear();
    for (auto& [direction, all_parent_vars] : *parent_value) {
      auto& parent_var = get<typename Subtag::tag>(all_parent_vars);
      auto& sub_var = (*sub_value)[direction];
      for (auto vars_it = parent_var.begin(), sub_var_it = sub_var.begin();
           vars_it != parent_var.end(); ++vars_it, ++sub_var_it) {
        sub_var_it->set_data_ref(&*vars_it);
      }
    }
  }

  // The `return_type` can be anything for Subitems because the DataBox figures
  // out the correct return type, we just use the `return_type` type alias to
  // signal to the DataBox we want mutating behavior.
  using return_type = NoSuchType;

  template <typename Subtag>
  static void create_compute_item(
      const gsl::not_null<typename Subtag::type*> sub_value,
      const typename tag::type& parent_value) noexcept {
    for (const auto& [direction, all_parent_vars] : parent_value) {
      const auto& parent_var = get<typename Subtag::tag>(all_parent_vars);
      auto& sub_var = (*sub_value)[direction];
      auto sub_var_it = sub_var.begin();
      for (auto vars_it = parent_var.begin(); vars_it != parent_var.end();
           ++vars_it, ++sub_var_it) {
        // clang-tidy: do not use const_cast
        // The DataBox will only give out a const reference to the
        // result of a compute item.  Here, that is a reference to a
        // const map to Tensors of DataVectors.  There is no (publicly
        // visible) indirection there, so having the map const will
        // allow only allow const access to the contained DataVectors,
        // so no modification through the pointer cast here is
        // possible.
        sub_var_it->set_data_ref(const_cast<DataVector*>(&*vars_it));  // NOLINT
      }
    }
  }
};
}  // namespace db
