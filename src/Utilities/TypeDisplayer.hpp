// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TypeDisplayer

#pragma once

/*!
 * \ingroup Utilities TypeTraits
 * \brief Get compiler error with type of template parameter
 *
 * The compiler error generated when using an object of type
 * `TypeDisplayer<...>` contains the types of the template parameters. This
 * effectively provides printf-debugging for metaprogramming. For example,
 * \code
 * TypeDisplayer<std::vector<double>> some_random_name;
 * \endcode
 * will produce a compiler error that contains the type `std::vector<double,
 * std::allocatior...>`. TypeDisplayer is extremely useful when debugging
 * template metaprograms.
 *
 * \note The TypeDisplayer header should only be included during testing
 * and debugging.
 */
template <typename...>
struct TypeDisplayer;
