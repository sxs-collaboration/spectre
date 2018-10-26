// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Raises or lowers the first index of a rank 3 tensor which is symmetric
 * in the last two indices.
 *
 * \details If \f$T_{abc}\f$ is a tensor with \f$T_{abc} = T_{acb}\f$ and the
 * indices \f$a,b,c,...\f$ can represent either spatial or spacetime indices,
 * then the tensor \f$ T^a_{bc} = g^{ad} T_{abc} \f$ is computed, where \f$
 * g^{ab}\f$ is the inverse metric, which is either a spatial or spacetime
 * metric. If a tensor \f$ S^a_{bc} \f$ is passed as an argument than the
 * corresponding tensor \f$ S_{abc} \f$ is calculated with respect to the metric
 * \f$g_{ab}\f$.  You may have to add a new instantiation of this template if
 * you need a new use case.
 */
template <typename DataType, typename Index0, typename Index1>
void raise_or_lower_first_index(
    gsl::not_null<
        Tensor<DataType, Symmetry<2, 1, 1>,
               index_list<change_index_up_lo<Index0>, Index1, Index1>>*>
        result,
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept;

template <typename DataType, typename Index0, typename Index1>
Tensor<DataType, Symmetry<2, 1, 1>,
       index_list<change_index_up_lo<Index0>, Index1, Index1>>
raise_or_lower_first_index(
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept {
  auto result = make_with_value<
      Tensor<DataType, Symmetry<2, 1, 1>,
             index_list<change_index_up_lo<Index0>, Index1, Index1>>>(metric,
                                                                      0.);
  raise_or_lower_first_index(make_not_null(&result), tensor, metric);
  return result;
}
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Raises or lowers the index of a rank 1 tensor.
 *
 * \details If \f$T_{a}\f$ is a tensor and the
 * index \f$a\f$ can represent either a spatial or spacetime index,
 * then the tensor \f$ T^a = g^{ad} T_{d} \f$ is computed, where \f$
 * g^{ab}\f$ is the inverse metric, which is either a spatial or spacetime
 * metric. If a tensor \f$ S^a \f$ is passed as an argument than the
 * corresponding tensor \f$ S_{a} \f$ is calculated with respect to the metric
 * \f$g_{ab}\f$.
 */
template <typename DataType, typename Index0>
void raise_or_lower_index(
    gsl::not_null<
        Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index0>>>*>
        result,
    const Tensor<DataType, Symmetry<1>, index_list<Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept;

template <typename DataType, typename Index0>
Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index0>>>
raise_or_lower_index(
    const Tensor<DataType, Symmetry<1>, index_list<Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept {
  auto result = make_with_value<
      Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index0>>>>(
      metric, 0.);
  raise_or_lower_index(make_not_null(&result), tensor, metric);
  return result;
}
// @}

// @{
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
                            change_index_up_lo<Index1>>>& metric) noexcept;

template <typename DataType, typename Index0, typename Index1>
Tensor<DataType, Symmetry<1>, index_list<Index0>> trace_last_indices(
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index1>,
                            change_index_up_lo<Index1>>>& metric) noexcept {
  auto trace_of_tensor =
      make_with_value<Tensor<DataType, Symmetry<1>, index_list<Index0>>>(metric,
                                                                         0.);
  trace_last_indices(make_not_null(&trace_of_tensor), tensor, metric);
  return trace_of_tensor;
}
// @}

// @{
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
                            change_index_up_lo<Index0>>>& metric) noexcept;

template <typename DataType, typename Index0>
Scalar<DataType> trace(
    const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept {
  Scalar<DataType> trace{};
  ::trace(make_not_null(&trace), tensor, metric);
  return trace;
}
// @}
