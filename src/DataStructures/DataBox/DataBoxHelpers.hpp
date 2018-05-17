// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper classes and functions for manipulating DataBox's and items

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/BoostHelpers.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;

class DataVector;
/// \endcond

template <typename Tag, typename = std::nullptr_t>
struct get_item_from_variant_databox;

template <typename Tag>
struct get_item_from_variant_databox<
    Tag, Requires<std::is_base_of<db::SimpleTag, Tag>::value>>
    : boost::static_visitor<db::item_type<Tag>> {
  template <typename DataBox_t>
  constexpr db::item_type<Tag> operator()(DataBox_t& box) const {
    return box.template get<Tag>();
  }
};

template <typename TagType>
struct get_item_from_variant_databox<
    TagType, Requires<not std::is_base_of<db::SimpleTag, TagType>::value>>
    : boost::static_visitor<TagType> {
  explicit get_item_from_variant_databox(std::string name)
      : var_name(std::move(name)) {}
  template <typename DataBox_t>
  constexpr TagType operator()(DataBox_t& box) const {
    return db::get_item_from_box<TagType>(box, var_name);
  }

 private:
  const std::string var_name;
};

namespace DataBoxHelpers_detail {
template <typename Tags, typename TagList,
          Requires<(tmpl::size<Tags>::value == 1)> = nullptr>
auto get_tensor_from_box(const db::DataBox<TagList>& box,
                         const std::string& tag_name) {
  using tag = tmpl::front<Tags>;
  if (db::get_tag_name<tag>() != tag_name) {
    ERROR("Could not find the tag named \"" << tag_name << "\" in the DataBox");
  }
  return ::db::get<tag>(box).get_vector_of_data();
}

template <typename Tags, typename TagList,
          Requires<(tmpl::size<Tags>::value > 1)> = nullptr>
auto get_tensor_from_box(const db::DataBox<TagList>& box,
                         const std::string& tag_name) {
  using tag = tmpl::front<Tags>;
  return db::get_tag_name<tag>() == tag_name
             ? ::db::get<tag>(box).serialize()
             : get_tensor_from_box<tmpl::pop_front<Tags>>(box, tag_name);
}

template <typename T>
using is_a_tensor = tt::is_a<Tensor, T>;
}  // namespace DataBoxHelpers_detail

template <typename TagsList>
auto get_tensor_from_box(const db::DataBox<TagsList>& box,
                         const std::string& tag_name) {
  using tags =
      tmpl::filter<TagsList, tmpl::bind<DataBoxHelpers_detail::is_a_tensor,
                                        tmpl::bind<db::item_type, tmpl::_1>>>;
  return DataBoxHelpers_detail::get_tensor_from_box<tags>(box, tag_name);
}

// namespace DataBoxHelpers_detail {
// template <typename Tags, typename TagsList,
//          Requires<(tmpl::size<Tags>::value == 1)> = nullptr>
// auto get_tensor_norm_from_box(
//    const db::DataBox<TagsList>& box,
//    const std::pair<std::string, TypeOfNorm>& tag_name) {
//  using tag = tmpl::front<Tags>;
//  if (db::get_tag_name<tag>() != tag_name.first) {
//    ERROR("Could not find the tag named \"" << tag_name.first
//                                            << "\" in the DataBox");
//  }
//  return compute_norm_core(box.template get<tag>(), tag_name.second);
//}
//
// template <typename Tags, typename TagsList,
//          Requires<(tmpl::size<Tags>::value > 1)> = nullptr>
// auto get_tensor_norm_from_box(
//    const db::DataBox<TagsList>& box,
//    const std::pair<std::string, TypeOfNorm>& tag_name) {
//  using tag = tmpl::front<Tags>;
//  if (db::get_tag_name<tag>() != tag_name.first) {
//    return get_tensor_norm_from_box<tmpl::pop_front<Tags>>(box, tag_name);
//  }
//  return compute_norm_core(box.template get<tag>(), tag_name.second);
//}
//}  // namespace DataBoxHelpers_detail

// template <typename TagsList>
// auto get_tensor_norm_from_box(
//    const db::DataBox<TagsList>& box,
//    const std::pair<std::string, TypeOfNorm>& tag_name) {
//  using tags =
//      tmpl::filter<TagsList, tmpl::bind<DataBoxHelpers_detail::is_a_tensor,
//                                      tmpl::bind<db::item_type, tmpl::_1>>>;
//
//  return DataBoxHelpers_detail::get_tensor_norm_from_box<tags>(box, tag_name);
//}

struct get_tensor_from_variant_box
    : boost::static_visitor<
          std::pair<std::vector<std::string>, std::vector<DataVector>>> {
  explicit get_tensor_from_variant_box(std::string name)
      : var_name(std::move(name)) {}
  template <typename DataBox_t>
  std::pair<std::vector<std::string>, std::vector<DataVector>> operator()(
      DataBox_t& box) const {
    return get_tensor_from_box(box, var_name);
  }

 private:
  const std::string var_name;
};
