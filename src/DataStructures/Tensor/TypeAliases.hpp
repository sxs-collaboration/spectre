// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines a list of useful type aliases for tensors

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

/// \ingroup TensorGroup
/// Scalar type
template <typename T>
using Scalar = Tensor<T, Symmetry<>, index_list<>>;

/*!
 * \ingroup TensorGroup
 * \brief Type aliases to construct common Tensors
 *
 * Lower case letters represent covariant indices and upper case letters
 * represent contravariant indices. Letters a, b, c, d represent spacetime
 * indices and i, j, k, l represent spatial indices.
 */
namespace tnsr {
// Rank 1
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using a = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo,
                                                           Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using A = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up,
                                                           Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using i = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using I = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                 index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;

// Rank 2
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using ab = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using Ab = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using aB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using AB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ij = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using Ij = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using IJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iA = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpacetimeIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ia = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;

template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using aa = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using AA = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ii = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using II = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1>,
                  index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                             SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;

// Rank 3 - spacetime
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using abc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using abC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using aBc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using Abc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using aBC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using AbC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using ABc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using ABC = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using abb = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using Abb = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using aBB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using ABB = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>>>;

// Rank 3 - spatial
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iii = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ijk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ijK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iJk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using Ijk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iJK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using IjK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using IJk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using IJK = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ijj = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using Ijj = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iJJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using IJJ = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using III = Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Up, Fr>>>;

// Rank 3 - mixed spacetime spatial
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iab = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iaB = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iAb = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iAB = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iaa = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using aia = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 2>,
                   index_list<SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iAA = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Up, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using Iaa = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                              SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;

// Rank 4 - Mixed
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ijaa = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1, 1>,
                    index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using Ijaa = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1, 1>,
                    index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpacetimeIndex<SpatialDim, UpLo::Lo, Fr>>>;

// Rank 4 - generic (default spacetime)
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using abcc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System,
          IndexType Index = IndexType::Spacetime>
using aBcc = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 3, 2, 1, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Up, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;

// Rank 4 - spatial
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using ijkk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1, 1>,
                    index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;
template <typename DataType, size_t SpatialDim, typename Fr = Frame::System>
using iJkk = Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1, 1>,
                    index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                               SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>;

}  // namespace tnsr

template <typename DataType, size_t Dim, typename SourceFrame,
          typename TargetFrame>
using InverseJacobian =
    Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<Dim, UpLo::Up, SourceFrame>,
                      SpatialIndex<Dim, UpLo::Lo, TargetFrame>>>;

template <typename DataType, size_t Dim, typename SourceFrame,
          typename TargetFrame>
using Jacobian = Tensor<DataType, tmpl::integral_list<std::int32_t, 2, 1>,
                        index_list<SpatialIndex<Dim, UpLo::Up, TargetFrame>,
                                   SpatialIndex<Dim, UpLo::Lo, SourceFrame>>>;
