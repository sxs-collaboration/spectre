// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Tags {
template <size_t N, typename T>
struct TempTensor : db::SimpleTag {
  using type = T;
  static std::string name() {
    return std::string("TempTensor") + std::to_string(N);
  }
};

/// @{
/// \ingroup PeoGroup
/// Variables Tags for temporary tensors inside a function.
template <size_t N, typename DataType = DataVector>
using TempScalar = TempTensor<N, Scalar<DataType>>;

// Rank 1
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempa = TempTensor<N, tnsr::a<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempA = TempTensor<N, tnsr::A<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempi = TempTensor<N, tnsr::i<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempI = TempTensor<N, tnsr::I<DataType, SpatialDim, Fr>>;

// Rank 2
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempab = TempTensor<N, tnsr::ab<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempaB = TempTensor<N, tnsr::aB<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempAb = TempTensor<N, tnsr::Ab<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempAB = TempTensor<N, tnsr::AB<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempij = TempTensor<N, tnsr::ij<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempiJ = TempTensor<N, tnsr::iJ<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempIj = TempTensor<N, tnsr::Ij<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempIJ = TempTensor<N, tnsr::IJ<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempia = TempTensor<N, tnsr::ia<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempaa = TempTensor<N, tnsr::aa<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempAA = TempTensor<N, tnsr::AA<DataType, SpatialDim, Fr>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempii = TempTensor<N, tnsr::ii<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempII = TempTensor<N, tnsr::II<DataType, SpatialDim, Fr>>;

// Rank 3
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempijj = TempTensor<N, tnsr::ijj<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempIjj = TempTensor<N, tnsr::Ijj<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempijk = TempTensor<N, tnsr::ijk<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempijK = TempTensor<N, tnsr::ijK<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempiaa = TempTensor<N, tnsr::iaa<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempIaa = TempTensor<N, tnsr::Iaa<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempiaB = TempTensor<N, tnsr::iaB<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempabb = TempTensor<N, tnsr::abb<DataType, SpatialDim, Fr>>;
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using TempabC = TempTensor<N, tnsr::abC<DataType, SpatialDim, Fr>>;

// Rank 4
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Tempijaa = TempTensor<N, tnsr::ijaa<DataType, SpatialDim, Fr>>;

// Temporary tensor types used in the file `ApparentHorizons/StrahlkorperGr.cpp`
template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Temp_hessian =
    TempTensor<N, Tensor<DataType, tmpl::integral_list<std::int32_t, 3, 2, 1>,
                         index_list<SpatialIndex<SpatialDim, UpLo::Up, Fr>,
                                    SpatialIndex<SpatialDim - 1, UpLo::Lo,
                                                 ::Frame::Spherical<Fr>>,
                                    SpatialIndex<SpatialDim - 1, UpLo::Lo,
                                                 ::Frame::Spherical<Fr>>>>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Temp_surface_dual_basis =
    TempTensor<N, Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 2>,
                         index_list<SpatialIndex<SpatialDim - 1, UpLo::Lo, Fr>,
                                    SpatialIndex<SpatialDim, UpLo::Lo, Fr>>>>;

template <size_t N, size_t SpatialDim, typename Fr = Frame::Inertial,
          typename DataType = DataVector>
using Temp_grad_spatial_metric =
    TempTensor<N,
               Tensor<DataType, tmpl::integral_list<std::int32_t, 1, 1, 2>,
                      index_list<SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                                 SpatialIndex<SpatialDim, UpLo::Lo, Fr>,
                                 SpatialIndex<SpatialDim - 1, UpLo::Lo, Fr>>>>;

/// @}
}  // namespace Tags
