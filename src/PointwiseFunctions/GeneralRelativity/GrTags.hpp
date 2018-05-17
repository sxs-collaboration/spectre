// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "GrTagsDeclarations.hpp"

namespace gr {
namespace Tags {
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeMetric : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
  static constexpr db::Label label = "SpacetimeMetric";
};
template <size_t Dim, typename Frame, typename DataType>
struct InverseSpacetimeMetric : db::SimpleTag {
  using type = tnsr::AA<DataType, Dim, Frame>;
  static constexpr db::Label label = "InverseSpacetimeMetric";
};

template <size_t Dim, typename Frame, typename DataType>
struct SpatialMetric : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
  static constexpr db::Label label = "SpatialMetric";
};
template <size_t Dim, typename Frame, typename DataType>
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataType, Dim, Frame>;
  static constexpr db::Label label = "InverseSpatialMetric";
};
template <size_t Dim, typename Frame, typename DataType>
struct SqrtDetSpatialMetric : db::SimpleTag {
  using type = Scalar<DataType>;
  static constexpr db::Label label = "SqrtDetSpatialMetric";
};
template <size_t Dim, typename Frame, typename DataType>
struct Shift : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
  static constexpr db::Label label = "Shift";
};
template <size_t Dim, typename Frame, typename DataType>
struct Lapse : db::SimpleTag {
  using type = Scalar<DataType>;
  static constexpr db::Label label = "Lapse";
};

template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::abb<DataType, Dim, Frame>;
  static constexpr db::Label label = "SpacetimeChristoffelFirstKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Abb<DataType, Dim, Frame>;
  static constexpr db::Label label = "SpactimeChristoffelSecondKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeNormalOneForm : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
  static constexpr db::Label label = "SpacetimeNormalOneForm";
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeNormalVector : db::SimpleTag {
  using type = tnsr::A<DataType, Dim, Frame>;
  static constexpr db::Label label = "SpacetimeNormalVector";
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpacetimeChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
  static constexpr db::Label label = "TraceSpacetimeChristoffelFirstKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpatialChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
  static constexpr db::Label label = "TraceSpatialChristoffelSecondKind";
};
template <size_t Dim, typename Frame, typename DataType>
struct ExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
  static constexpr db::Label label = "ExtrinsicCurvature";
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceExtrinsicCurvature : db::SimpleTag {
  using type = Scalar<DataType>;
  static constexpr db::Label label = "TraceExtrinsicCurvature";
};
}  // namespace Tags
}  // namespace gr
