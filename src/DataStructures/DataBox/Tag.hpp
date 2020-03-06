// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace db {
/*!
 * \ingroup DataBoxGroup
 * \brief Tags for the DataBox inherit from this type
 *
 * \details
 * Used to mark a type as being a SimpleTag so that it can be used in a
 * DataBox.
 *
 * \derivedrequires
 * - type alias `type` of the type this SimpleTag represents
 *
 * \example
 * \snippet Test_DataBox.cpp databox_tag_example
 *
 * \see DataBox PrefixTag tag_name
 */
struct SimpleTag {};

/*!
 * \ingroup DataBoxGroup
 * \brief Tags that are base tags, i.e. a simple or compute tag must derive
 * off them for them to be useful
 *
 * Base tags do not need to contain type information, unlike simple
 * tags which must contain the type information. Base tags are designed so
 * that retrieving items from the DataBox or setting argument tags in compute
 * items can be done without any knowledge of the type of the item.
 *
 * To use the base mechanism the base tag must inherit off of
 * `BaseTag` and NOT `SimpleTag`. This is very important for the
 * implementation. Inheriting off both and not making the tag either a simple
 * item or compute item is undefined behavior and is likely to end in extremely
 * complicated compiler errors.
 */
struct BaseTag {};

/*!
 * \ingroup DataBoxGroup
 * \brief Marks an item as being a prefix to another tag
 *
 * \details
 * Used to mark a type as being a DataBoxTag where the `label` is a prefix to
 * the DataBoxTag that is a member type alias `tag`. A prefix tag must contain a
 * type alias named `type` with the type of the Tag it is a prefix to, as well
 * as a type alias `tag` that is the type of the Tag that this prefix tag is
 * a prefix for. A prefix tag must also have a `label` equal to the name of
 * the struct (tag).
 *
 * \derivedrequires
 * - type alias `tag` of the DataBoxTag that this tag is a prefix to
 * - type alias `type` that is the type that this PrefixTag holds
 * - static `std::string name()` method that returns a runtime name for the tag.
 *
 * \example
 * A PrefixTag tag has the structure:
 * \snippet Test_DataBox.cpp databox_prefix_tag_example
 *
 * The name used to retrieve a prefix tag from the DataBox is:
 * \snippet Test_DataBox.cpp databox_name_prefix
 *
 *
 * \see DataBox DataBoxTag tag_name ComputeTag
 */
struct PrefixTag {};

/*!
 * \ingroup DataBoxGroup
 * \brief Marks a DataBoxTag as being a compute item that executes a function
 *
 * \details
 * Compute items come in two forms: mutating and non-mutating. Mutating
 * compute items modify a stored value in order to reduce the number of memory
 * allocations done. For example, if a function would return a `Variables` or
 * `Tensor<DataVector...>` and is called every time step, then it would be
 * preferable to use a mutating compute item so that the values in the already
 * allocated memory can just be changed.
 * In contrast, non-mutating compute items simply return the new value after a
 * call (if the value is out-of-date), which is fine for infrequently called
 * compute items or ones that do not allocate data on the heap.
 *
 * A compute item tag contains a member named `function` that is either a
 * function pointer, or a static constexpr function. The compute item tag
 * must also have a `label`, same as the DataBox tags, and a type alias
 * `argument_tags` that is a typelist of the tags that will
 * be retrieved from the DataBox and whose data will be passed to the function
 * (pointer). Mutating compute item tags must also contain a type alias named
 * `return_type` that is the type the function is mutating. The type must be
 * default constructible.
 *
 * \example
 * Most non-mutating compute item tags will look similar to:
 * \snippet Test_DataBox.cpp databox_compute_item_tag_example
 * Note that the arguments can be empty:
 * \snippet Test_DataBox.cpp compute_item_tag_no_tags
 *
 * Mutating compute item tags are of the form:
 * \snippet Test_DataBox.cpp databox_mutating_compute_item_tag
 * where the function is:
 * \snippet Test_DataBox.cpp databox_mutating_compute_item_function
 *
 * You can also have `function` be a function instead of a function pointer,
 * which offers a lot of simplicity for very simple compute items.
 * \snippet Test_DataBox.cpp compute_item_tag_function
 *
 * \see DataBox SimpleTag tag_name PrefixTag
 */
struct ComputeTag {};

}  // namespace db
