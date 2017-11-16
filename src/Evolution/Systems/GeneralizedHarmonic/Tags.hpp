// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace GeneralizedHarmonic {
template <size_t Dim>
struct SpacetimeMetric : db::DataBoxTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString_t label = "SpacetimeMetric";
};

/*!
 * \brief Conjugate momentum to the spacetime metric.
 *
 * \details If \f$ \psi_{ab} \f$ is the spacetime metric, and \f$ N \f$ and
 * \f$ N^i \f$ are the lapse and shift respectively, then we define
 * \f$ \Pi_{ab} = -\frac{1}{N} ( \partial_t \psi_{ab} + N^{i} \phi_{iab} ) \f$
 * where \f$\phi_{iab}\f$ is the variable defined by the tag Phi.
 */
template <size_t Dim>
struct Pi : db::DataBoxTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString_t label = "Pi";
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * spacetime metric
 * \details If \f$\psi_{ab}\f$ is the spacetime metric then we define
 * \f$\phi_{iab} = \partial_i \psi_{ab}\f$
 */
template <size_t Dim>
struct Phi : db::DataBoxTag {
  using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString_t label = "Phi";
};

template <size_t Dim>
struct InverseSpatialMetric : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "InverseSpatialMetric";
  // static constexpr auto function =
  // compute_inverse_spatial_metric_from_spacetime_metric<Dim,
  //      Frame::Grid>;
  using argument_tags = typelist<SpacetimeMetric<Dim>>;
};
template <size_t Dim>
struct Shift : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "Shift";
  // static constexpr auto function =
  //      compute_shift_from_spacetime_metric_and_invg<Dim, Frame::Grid>;
  using argument_tags =
      typelist<SpacetimeMetric<Dim>, InverseSpatialMetric<Dim>>;
};
template <size_t Dim>
struct Lapse : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "Lapse";
  // static constexpr auto function =
  //     compute_lapse_from_spacetime_metric_shift<Dim, Frame::Grid>;
  using argument_tags = typelist<SpacetimeMetric<Dim>, Shift<Dim>>;
};
struct ConstraintGamma0 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "ConstraintGamma0";
};
struct ConstraintGamma1 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "ConstraintGamma1";
};
struct ConstraintGamma2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "ConstraintGamma2";
};
template <size_t Dim>
struct GaugeH : db::DataBoxTag {
  using type = tnsr::a<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString_t label = "GaugeH";
};
template <size_t Dim>
struct SpacetimeDerivGaugeH : db::DataBoxTag {
  using type = tnsr::ab<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString_t label = "SpacetimeDerivGaugeH";
};
template <size_t Dim>
struct InverseSpacetimeMetric : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "InverseSpacetimeMetric";
  // static constexpr auto function =
  //     compute_inverse_spacetime_metric_from_invg_lapse_shift<Dim,
  //     Frame::Grid>;
  using argument_tags =
      typelist<InverseSpatialMetric<Dim>, Shift<Dim>, Lapse<Dim>>;
};
template <size_t Dim>
struct SpacetimeChristoffelFirstKind : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "SpacetimeChristoffelFirstKind";
  // static constexpr auto function =
  //     compute_christoffel_first_kind_from_gh<Dim, Frame::Grid>;
  using argument_tags =
      typelist<InverseSpacetimeMetric<Dim>, Pi<Dim>, Phi<Dim>>;
};
template <size_t Dim>
struct SpacetimeChristoffelSecondKind : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "SpactimeChristoffelSecondKind";
  // static constexpr auto function =
  //     raise_index_1_of_3<Dim, Frame::Grid, IndexType::Spacetime>;
  using argument_tags =
      typelist<SpacetimeChristoffelFirstKind<Dim>, InverseSpacetimeMetric<Dim>>;
};
template <size_t Dim>
struct SpacetimeNormalOneForm : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "SpacetimeNormalOneForm";
  // static constexpr auto function =
  //     compute_normal_one_form_from_lapse<Dim, Frame::Grid>;
  using argument_tags = typelist<Lapse<Dim>>;
};
template <size_t Dim>
struct SpacetimeNormalVector : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "SpacetimeNormalVector";
  // static constexpr auto function =
  //     compute_normal_vector_from_lapse_and_shift<Dim, Frame::Grid>;
  using argument_tags = typelist<Lapse<Dim>, Shift<Dim>>;
};
template <size_t Dim>
struct TraceSpacetimeChristoffelFirstKind : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label =
      "TraceSpacetimeChristoffelFirstKind";
  // static constexpr auto function =
  //     compute_trace_23_of_3<Dim, Frame::Grid, IndexType::Spacetime>;
  using argument_tags =
      typelist<SpacetimeChristoffelFirstKind<Dim>, InverseSpacetimeMetric<Dim>>;
};
}  // namespace GeneralizedHarmonic
