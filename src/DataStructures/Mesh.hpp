// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Mesh.

#pragma once

#include "DataStructures/Index.hpp"

/*!
 * \ingroup DataStructures
 * \brief Represents the size of a `Dim` dimensional mesh as `Dim` integers
 * \see Index
 */
template <size_t Dim>
using Mesh = Index<Dim>;
