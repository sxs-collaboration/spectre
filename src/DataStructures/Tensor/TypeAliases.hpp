// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines a list of useful type aliases for tensors

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"

class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;

/// \ingroup Tensor
/// Scalar type
template <typename T>
using Scalar = Tensor<T, Symmetry<>, index_list<>>;



/*!
 * \ingroup Tensor
 * \brief Type aliases to construct common Tensors
 *
 * Lower case letters represent covariant indices and upper case letters
 * represent contravariant indices. Letters a, b, c, d represent spacetime
 * indices and i, j, k, l represent spatial indices.
 */
namespace tnsr {
// Rank 1
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using a = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo,
                                                           Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using A = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up,
                                                           Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using i = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using I = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;

// Rank 2
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using ab = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using Ab = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using aB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using AB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using ij = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using Ij = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using IJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;

template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using aa = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using AA = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using ii = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using II = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;

// Rank 3 - spacetime
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using abc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using abC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using aBc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using Abc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using aBC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using AbC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using ABc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using ABC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using abb = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using Abb = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using aBB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr,
          IndexType Index = IndexType::Spacetime>
using ABB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;

// Rank 3 - spatial
template <typename DataType, size_t SpatialDim, typename Fr>
using iii = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using ijk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using ijK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iJk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using Ijk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iJK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using IjK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using IJk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using IJK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using ijj = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using Ijj = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iJJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using IJJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using III = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;

// Rank 3 - mixed spacetime spatial
template <typename DataType, size_t SpatialDim, typename Fr>
using iab = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iaB = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iAb = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iAB = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iaa = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr>
using iAA = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>>>;

namespace detail {
template <size_t Dim, typename Frame1, typename Frame2>
struct inverse_jacobian_impl {
  static_assert(tmpl::index_of<Frame::ordered_frame_list, Frame1>::value <
                    tmpl::index_of<Frame::ordered_frame_list, Frame2>::value,
                "Inverse Jacobian must go other direction.");
  using type = Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                      typelist<SpatialIndex<Dim, UpLo::Up, Frame1>,
                               SpatialIndex<Dim, UpLo::Lo, Frame2>>>;
};
}  // namespace detail

}  // namespace tnsr

template <size_t Dim, typename Frame1, typename Frame2>
using InverseJacobian =
    typename tnsr::detail::inverse_jacobian_impl<Dim, Frame1, Frame2>::type;

template <size_t SpatialDim, typename Fr>
using Point = Tensor<double, tmpl::integral_list<std::int32_t, 1>,
                     index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
