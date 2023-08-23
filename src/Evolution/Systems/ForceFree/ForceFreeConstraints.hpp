// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {

/*!
 * \brief Computes the scalar self-product $V^2 = V_i V^i$ where $V^i$ is either
 * $\tilde{E}^i$ or $\tilde{B}^i$.
 */
void tilde_e_or_b_squared(
    const gsl::not_null<Scalar<DataVector>*> tilde_e_or_b_squared,
    const tnsr::I<DataVector, 3>& densitized_vector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);

/*!
 * \brief Computes the scalar product $\tilde{E}_i \tilde{B}^i$.
 */
void tilde_e_dot_tilde_b(
    const gsl::not_null<Scalar<DataVector>*> tilde_e_dot_tilde_b,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);

/*!
 * \brief Computes the scalar product $E_iB^i = \tilde{E}_i \tilde{B}^i /
 * \gamma$, which measures the violation of the $E_iB^i = 0$ force-free
 * condition.
 */
void electric_field_dot_magnetic_field(
    const gsl::not_null<Scalar<DataVector>*> electric_field_dot_magnetic_field,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);

/*!
 * \brief Computes the violation of the magnetic dominance ($E^2 < B^2$)
 * force-free condition.
 */
void magnetic_dominance_violation(
    const gsl::not_null<Scalar<DataVector>*> magnetic_dominance_violation,
    const Scalar<DataVector>& tilde_e_squared,
    const Scalar<DataVector>& tilde_b_squared,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

namespace Tags {

/*!
 * \brief Computes $\tilde{E}^2 = \tilde{E}_i \tilde{E}^i$.
 */
struct TildeESquaredCompute : TildeESquared, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeE, gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = Scalar<DataVector>;
  using base = TildeESquared;

  static constexpr auto function = &tilde_e_or_b_squared;
};

/*!
 * \brief Computes $\tilde{B}^2 = \tilde{B}_i \tilde{B}^i$.
 */
struct TildeBSquaredCompute : TildeBSquared, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeB, gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = Scalar<DataVector>;
  using base = TildeBSquared;

  static constexpr auto function = &tilde_e_or_b_squared;
};

/*!
 * \brief Computes the product $\tilde{E}_i\tilde{B}^i$.
 */
struct TildeEDotTildeBCompute : TildeEDotTildeB, db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeE, TildeB, gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = Scalar<DataVector>;
  using base = TildeEDotTildeB;

  static constexpr auto function = &tilde_e_dot_tilde_b;
};

/*!
 * \brief Computes the dot product of electric and magnetic fields \f$E^iB_i\f$.
 *
 * This quantity must vanish in the ideal force-free limit.
 *
 */
struct ElectricFieldDotMagneticFieldCompute : ElectricFieldDotMagneticField,
                                              db::ComputeTag {
  using argument_tags =
      tmpl::list<TildeE, TildeB, gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = Scalar<DataVector>;
  using base = ElectricFieldDotMagneticField;

  static constexpr auto function = &electric_field_dot_magnetic_field;
};

/*!
 * \brief Computes
 *
 */
struct MagneticDominanceViolationCompute : MagneticDominanceViolation,
                                           db::ComputeTag {
  using argument_tags = tmpl::list<TildeESquared, TildeBSquared,
                                   gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using return_type = Scalar<DataVector>;
  using base = MagneticDominanceViolation;

  static constexpr auto function = &magnetic_dominance_violation;
};

}  // namespace Tags
}  // namespace ForceFree
