// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes trace of a rank 3 tensor, which is symmetric in its last two
 * indices, tracing the symmetric indices.
 *
 * \details For example, if \f$ T_{abc} \f$ is a tensor such that \f$T_{abc} =
 * T_{acb} \f$ then \f$ T_a = g^{bc}T_{abc} \f$ is computed, where \f$ g^{bc}
 * \f$ is the inverse metric.  Note that indices \f$a,b,c,...\f$ can represent
 * either spatial or spacetime indices, and can have either valence.  You may
 * have to add a new instantiation of this template if you need a new use case.
 */
template <typename DataType, typename Index0, typename Index1>
void trace_last_indices(
    gsl::not_null<Tensor<DataType, Symmetry<1>, index_list<Index0>>*>
        trace_of_tensor,
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index1>,
                            change_index_up_lo<Index1>>>& metric);

template <typename DataType, typename Index0, typename Index1>
Tensor<DataType, Symmetry<1>, index_list<Index0>> trace_last_indices(
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index1>,
                            change_index_up_lo<Index1>>>& metric) {
  auto trace_of_tensor =
      make_with_value<Tensor<DataType, Symmetry<1>, index_list<Index0>>>(metric,
                                                                         0.);
  trace_last_indices(make_not_null(&trace_of_tensor), tensor, metric);
  return trace_of_tensor;
}
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes trace of a rank-2 symmetric tensor.
 * \details Computes \f$g^{ab}T_{ab}\f$ or \f$g_{ab}T^{ab}\f$ where \f$(a,b)\f$
 * can be spatial or spacetime indices.
 */
template <typename DataType, typename Index0>
void trace(
    gsl::not_null<Scalar<DataType>*> trace,
    const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric);

template <typename DataType, typename Index0>
Scalar<DataType> trace(
    const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  Scalar<DataType> trace{};
  ::trace(make_not_null(&trace), tensor, metric);
  return trace;
}
/// @}
