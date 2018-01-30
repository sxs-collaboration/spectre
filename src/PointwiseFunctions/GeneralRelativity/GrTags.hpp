// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "GrTagsDeclarations.hpp"

namespace gr {
namespace Tags {
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeMetric : db::DataBoxTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "SpacetimeMetric";
};
template <size_t Dim, typename Frame, typename DataType>
struct InverseSpacetimeMetric : db::DataBoxTag {
  using type = tnsr::AA<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "InverseSpacetimeMetric";
};

template <size_t Dim, typename Frame, typename DataType>
struct SpatialMetric : db::DataBoxTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "SpatialMetric";
};
template <size_t Dim, typename Frame, typename DataType>
struct InverseSpatialMetric : db::DataBoxTag {
  using type = tnsr::II<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "InverseSpatialMetric";
};
template <size_t Dim, typename Frame, typename DataType>
struct Shift : db::DataBoxTag {
  using type = tnsr::I<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "Shift";
};
template <size_t Dim, typename Frame, typename DataType>
struct Lapse : db::DataBoxTag {
  using type = Scalar<DataType>;
  static constexpr db::DataBoxString label = "Lapse";
};

template <size_t Dim, typename Frame, typename DataType>
struct DtShift : db::DataBoxTag {
  using type = tnsr::I<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "DtShift";
};
template <size_t Dim, typename Frame, typename DataType>
struct DtLapse : db::DataBoxTag {
  using type = Scalar<DataType>;
  static constexpr db::DataBoxString label = "DtLapse";
};
template <size_t Dim, typename Frame, typename DataType>
struct DtSpatialMetric : db::DataBoxTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "DtSpatialMetric";
};

template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeChristoffelFirstKind : db::DataBoxTag {
  using type = tnsr::abb<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "SpacetimeChristoffelFirstKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeChristoffelSecondKind : db::DataBoxTag {
  using type = tnsr::Abb<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "SpactimeChristoffelSecondKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeNormalOneForm : db::DataBoxTag {
  using type = tnsr::a<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "SpacetimeNormalOneForm";
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeNormalVector : db::DataBoxTag {
  using type = tnsr::A<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label = "SpacetimeNormalVector";
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpacetimeChristoffelFirstKind : db::DataBoxTag {
  using type = tnsr::a<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label =
      "TraceSpacetimeChristoffelFirstKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpatialChristoffelSecondKind : db::DataBoxTag {
  using type = tnsr::I<DataType, Dim, Frame>;
  static constexpr db::DataBoxString label =
      "TraceSpatialChristoffelSecondKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceExtrinsicCurvature : db::DataBoxTag {
  using type = Scalar<DataType>;
  static constexpr db::DataBoxString label = "TraceExtrinsicCurvature";
};
}  // namespace Tags
}  // namespace gr
