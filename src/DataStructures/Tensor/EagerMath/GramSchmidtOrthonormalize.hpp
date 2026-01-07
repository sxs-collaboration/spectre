// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/*!
 * \brief Orthonormalize a set of vectors using the modified Gram-Schmidt
 * process.
 *
 * The first vector in `basis` is normalized, then each subsequent vector is
 * orthogonalized against all previous vectors and then normalized.
 */
template <typename DataType, typename Index, size_t NumVectors>
void gram_schmidt_orthonormalize(
    const std::array<
        gsl::not_null<Tensor<DataType, Symmetry<1>, index_list<Index>>*>,
        NumVectors>& basis,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric);
