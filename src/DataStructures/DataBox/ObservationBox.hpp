// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Item.hpp"
#include "Utilities/ErrorHandling/StaticAssert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename ComputeTagsList, typename DataBoxType>
class ObservationBox;
/// \endcond

namespace Tags {
/*!
 * \brief Tag used to retrieve the ObservationBox from the `get()` function
 *
 * The main use of this tag is to allow fetching the ObservationBox from itself.
 * The intended primary use case is for Events to be able to retrieve the
 * ObservationBox and then do runtime retrieval of tags to avoid computing
 * quantities that aren't needed.
 */
struct ObservationBox {
  // Trick to get friend function declaration to compile but a const
  // NoSuchtype****& is rather useless
  using type = NoSuchType****;
};
}

/*!
 * \ingroup DataStructuresGroup
 * \brief Used for adding compute items to a `DataBox` without copying or moving
 * any data from the original `DataBox`
 *
 * The intended use-case for this class is during IO/observing where additional
 * compute tags are needed only for observation. The memory used by those
 * compute tags does not need to be persistent and so we'd like a light-weight
 * class to handle the on-demand computation.
 */
template <typename DataBoxType, typename... ComputeTags>
class ObservationBox<tmpl::list<ComputeTags...>, DataBoxType>
    : private db::detail::Item<ComputeTags>... {
 public:
  /// A list of all the compute item tags
  using compute_item_tags = tmpl::list<ComputeTags...>;
  /// A list of all tags
  using tags_list =
      tmpl::push_back<typename DataBoxType::tags_list, ComputeTags...>;

  ObservationBox() = default;
  ObservationBox(const ObservationBox& rhs) = default;
  ObservationBox& operator=(const ObservationBox& rhs) = default;
  ObservationBox(ObservationBox&& rhs) = default;
  ObservationBox& operator=(ObservationBox&& rhs) = default;
  ~ObservationBox() = default;

  /// Create an `ObservationBox` that can also retrieve things out of the
  /// `databox` passed in.
  ObservationBox(const DataBoxType& databox);

  /// Retrieve the tag `Tag`, should be called by the free function db::get
  template <typename Tag>
  const auto& get() const;

 private:
  template <typename Tag>
  const auto& get_item() const {
    return static_cast<const db::detail::Item<Tag>&>(*this);
  }

  template <typename ComputeTag, typename... ArgumentTags>
  void evaluate_compute_item(
      tmpl::list<ArgumentTags...> /*meta*/) const;

  const DataBoxType* databox_ = nullptr;
};

/*!
 * \ingroup DataStructuresGroup
 * \brief Retrieve a `Tag` from the `ObservationBox`.
 */
template <typename Tag, typename DataBoxType, typename... ComputeTags>
const auto& get(
    const ObservationBox<tmpl::list<ComputeTags...>, DataBoxType>& box) {
  return box.template get<Tag>();
}

template <typename DataBoxType, typename... ComputeTags>
ObservationBox<tmpl::list<ComputeTags...>, DataBoxType>::ObservationBox(
    const DataBoxType& databox)
    : databox_(&databox) {
  DEBUG_STATIC_ASSERT(
      (db::is_immutable_item_tag_v<ComputeTags> and ...),
      "All tags passed to ObservationBox must be compute tags.");
}

template <typename DataBoxType, typename... ComputeTags>
template <typename Tag>
const auto& ObservationBox<tmpl::list<ComputeTags...>, DataBoxType>::get()
    const {
  if constexpr (std::is_same_v<Tag, ::Tags::DataBox>) {
    return *databox_;
  } else if constexpr (std::is_same_v<Tag, ::Tags::ObservationBox>) {
    return *this;
  } else {
    DEBUG_STATIC_ASSERT(
        not db::detail::has_no_matching_tag_v<tags_list, Tag>,
        "Found no tags in the ObservationBox that match the tag "
        "being retrieved.");
    DEBUG_STATIC_ASSERT(
        db::detail::has_unique_matching_tag_v<tags_list, Tag>,
        "Found more than one tag in the ObservationBox that matches the tag "
        "being retrieved. This happens because more than one tag with the same "
        "base (class) tag was added to the ObservationBox, or because you add "
        "a compute tag that is already in the DataBox.");

    if constexpr (db::tag_is_retrievable_v<Tag, DataBoxType>) {
      return db::get<Tag>(*databox_);
    } else {
      using item_tag = db::detail::first_matching_tag<compute_item_tags, Tag>;
      if constexpr (db::detail::Item<item_tag>::item_type ==
                    db::detail::ItemType::Reference) {
        return item_tag::get(get<typename item_tag::parent_tag>());
      } else {
        if (not get_item<item_tag>().evaluated()) {
          evaluate_compute_item<item_tag>(typename item_tag::argument_tags{});
        }
        if constexpr (tt::is_a_v<std::unique_ptr, typename item_tag::type>) {
          return *(get_item<item_tag>().get());
        } else {
          return get_item<item_tag>().get();
        }
      }
    }
  }
}

template <typename DataBoxType, typename... ComputeTags>
template <typename ComputeTag, typename... ArgumentTags>
void ObservationBox<tmpl::list<ComputeTags...>, DataBoxType>::
    evaluate_compute_item(tmpl::list<ArgumentTags...> /*meta*/) const {
  get_item<ComputeTag>().evaluate(get<ArgumentTags>()...);
}

template <typename ComputeTagsList, typename DataBoxType>
auto make_observation_box(const DataBoxType& databox) {
  return ObservationBox<db::detail::expand_subitems<ComputeTagsList>,
                        DataBoxType>{databox};
}

namespace observation_box_detail {
template <typename DataBoxType, typename ComputeTagsList, typename... Args,
          typename F, typename... ArgumentTags>
auto apply(F&& f, tmpl::list<ArgumentTags...> /*meta*/,
           const ObservationBox<ComputeTagsList, DataBoxType>& observation_box,
           Args&&... args) {
  return std::forward<F>(f)(get<ArgumentTags>(observation_box)...,
                            std::forward<Args>(args)...);
}
}  // namespace observation_box_detail

/*!
 * \ingroup DataStructuresGroup
 * \brief Apply the function object `f` using its nested `argument_tags` list of
 * tags.
 */
template <typename DataBoxType, typename ComputeTagsList, typename... Args,
          typename F>
auto apply(F&& f,
           const ObservationBox<ComputeTagsList, DataBoxType>& observation_box,
           Args&&... args) {
  return observation_box_detail::apply(
      std::forward<F>(f), typename std::decay_t<F>::argument_tags{},
      observation_box, std::forward<Args>(args)...);
}
