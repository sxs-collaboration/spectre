// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
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
 * \brief Computes the non-stiff part \f$\tilde{J}^i_\mathrm{drift}\f$ of the
 * generalized electric current density \f$\tilde{J}^i\f$.
 *
 * \f{align}
 *  \tilde{J}^i_\mathrm{drift}
 *    = \alpha \sqrt{\gamma} q \frac{\epsilon^{ijk}_{(3)}E_jB_k}{B_lB^l}
 * \f}
 *
 * where \f$\alpha\f$ is lapse, \f$\gamma\f$ is the determinant of the spatial
 * metric, \f$q\f$ is charge density, \f$\epsilon^{ijk}_{(3)}\f$ is the spatial
 * Levi-Civita tensor, \f$E^i\f$ is the electric field, and \f$B^i\f$ is the
 * magnetic field.
 *
 */
struct ComputeDriftTildeJ {
  using argument_tags =
      tmpl::list<Tags::TildeQ, Tags::TildeE, Tags::TildeB,
                 Tags::ParallelConductivity, gr::Tags::Lapse<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = tnsr::I<DataVector, 3>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> drift_tilde_j,
      const Scalar<DataVector>& tilde_q,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      double parallel_conductivity, const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);
};

/*!
 * \brief Computes the stiff part \f$\tilde{J}^i_\mathrm{parallel}\f$ of the
 * generalized electric current density \f$\tilde{J}^i\f$.
 *
 * \f{align*}
 *  \tilde{J}^i_\mathrm{parallel}
 *    & = \alpha \sqrt{\gamma} \eta \left[
 *        \frac{(E_lB^l)B^i}{B^2} + \frac{\mathcal{R}(E^2-B^2)}{B^2} E^i
 *      \right]
 * \f}
 *
 * where \f$\alpha\f$ is lapse, \f$\gamma\f$ is the determinant of the spatial
 * metric, \f$E^i\f$ is the electric field, \f$B^i\f$ is the magnetic field,
 * \f$\eta\f$ is the parallel conductivity, and \f$\mathcal{R}(x) = \max
 * (x,0)\f$ is the rectifier function.
 *
 */
struct ComputeParallelTildeJ {
  using argument_tags =
      tmpl::list<Tags::TildeQ, Tags::TildeE, Tags::TildeB,
                 Tags::ParallelConductivity, gr::Tags::Lapse<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = tnsr::I<DataVector, 3>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> parallel_tilde_j,
      const Scalar<DataVector>& tilde_q,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      double parallel_conductivity, const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);
};

namespace Tags {
/*!
 * \brief Computes the densitized electric current density \f$\tilde{J}^i\f$.
 *
 * \f{align}
 *  \tilde{J}^i = \tilde{J}^i_\mathrm{drift} + \tilde{J}^i_\mathrm{parallel}
 *   = \alpha \left[
 *      \tilde{q} \frac{\epsilon^{ijk}_{(3)}\tilde{E}_j \tilde{B}_k}
 *                     {\tilde{B}_l \tilde{B}^l}
 *      + \eta \left\{
 *        \frac{(\tilde{E}_l\tilde{B}^l)\tilde{B}^i}{\tilde{B}^2}
 *      + \frac{\mathcal{R}(\tilde{E}^2-\tilde{B}^2)}{\tilde{B}^2} \tilde{E}^i
 *      \right\}
 * \right]
 * \f}
 *
 * See ComputeDriftTildeJ and ComputeParallelTildeJ for the details of each
 * terms.
 *
 */
struct ComputeTildeJ : TildeJ, db::ComputeTag {
  using argument_tags =
      tmpl::list<Tags::TildeQ, Tags::TildeE, Tags::TildeB,
                 Tags::ParallelConductivity, gr::Tags::Lapse<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = tnsr::I<DataVector, 3>;
  using base = TildeJ;

  static void function(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
      const Scalar<DataVector>& tilde_q,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      double parallel_conductivity, const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);
};
}  // namespace Tags

}  // namespace ForceFree
