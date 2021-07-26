// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace CurvedScalarWave::BoundaryCorrections {
/*!
 * \brief Computes the upwind multipenalty boundary correction for scalar wave
 * in curved spacetime.
 *
 * \details This implements the upwind multipenalty boundary correction term.
 * The general form is given by:
 *
 * \f[
 * G = T^{\rm ext}\Lambda^{\rm ext,-} {T^{\rm ext}}^{-1} U^{\rm ext}
 *      + T^{\rm int}\Lambda^{\rm int,+} {T^{\rm int}}^{-1} U^{\rm int}
 *   = T^{\rm ext}\Lambda^{\rm ext,-} V^{\rm ext}
 *      + T^{\rm int}\Lambda^{\rm int,+} V^{\rm int},
 * \f]
 *
 * where
 *
 *  - \f$G\f$ is a vector of numerical upwind fluxes dotted with the interface
 *       normal for all evolved variables;
 *  - \f$U\f$ is a vector of all evolved variables;
 *  - \f$T\f$ is a matrix whose columns are the eigenvectors of the
 *       characteristic matrix for the evolution system. It maps the
 *       evolved variables to characteristic variables \f$V\f$, s.t.
 *       \f$V := T^{-1}\cdot U\f$; and
 *  - \f$\Lambda^\pm\f$ are diagonal matrices containing positive / negative
 *       eigenvalues of \f$T\f$ as its diagonal elements, with the rest set to
 *       \f$ 0\f$.
 *
 * The superscripts \f${\rm int}\f$ and \f${\rm ext}\f$ indicate that the
 * corresponding variable at the element interface comes from the _interior_ or
 * _exterior_ of the element. Exterior of the element is naturally the interior
 * of its neighboring element. The sign of characteristic speeds indicate the
 * direction of propagation of the corresponding characteristic field with
 * respect to the interface normal that the field has been computed along, with
 * negative speeds indicating incoming characteristics, and positive speeds
 * indicating outgoing characteristics. The expressions implemented here differ
 * from Eq.(63) of \cite Teukolsky2015ega as that boundary term does not
 * consistently treat both sides of the interface on the same footing. Unlike in
 * \cite Teukolsky2015ega, in code the interface normal vector on the interior
 * side and the one on the exterior side point in opposite directions, and the
 * characteristic speeds end up having different signs. This class therefore
 * computes:
 *
 * \f[
 * G = T^{\rm ext}w_{\rm ext} V^{\rm ext} - T^{\rm int}w_{\rm int} V^{\rm int},
 * \f]
 *
 * with weights \f$w_{\rm ext} = -\Theta(\Lambda^{\rm ext})\cdot\Lambda^{\rm
 * ext}\f$, and \f$w_{\rm int} = \Theta(-\Lambda^{\rm int})\cdot\Lambda^{\rm
 * int}\f$, where \f$\Theta\f$ is the Heaviside function centered at zero,
 * \f$\Lambda = \Lambda^+ + \Lambda^-\f$, and the dot operator \f$(\cdot)\f$
 * indicates an element-wise product.
 */
template <size_t Dim>
class UpwindPenalty final : public BoundaryCorrection<Dim> {
 private:
  struct CharSpeedsTensor : db::SimpleTag {
    using type = tnsr::a<DataVector, 3, Frame::Inertial>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the UpwindPenalty boundary correction term for the scalar wave "
      "system in curved spacetime."};

  UpwindPenalty() = default;
  UpwindPenalty(const UpwindPenalty&) = default;
  UpwindPenalty& operator=(const UpwindPenalty&) = default;
  UpwindPenalty(UpwindPenalty&&) = default;
  UpwindPenalty& operator=(UpwindPenalty&&) = default;
  ~UpwindPenalty() override = default;

  /// \cond
  explicit UpwindPenalty(CkMigrateMessage* msg) noexcept;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(UpwindPenalty);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection<Dim>> get_clone() const noexcept override;

  using dg_package_field_tags =
      tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus,
                 Tags::ConstraintGamma2,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>,
                 CharSpeedsTensor>;
  using dg_package_data_temporary_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      Tags::ConstraintGamma1, Tags::ConstraintGamma2>;
  using dg_package_data_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_v_psi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> packaged_v_zero,
      gsl::not_null<Scalar<DataVector>*> packaged_v_plus,
      gsl::not_null<Scalar<DataVector>*> packaged_v_minus,
      gsl::not_null<Scalar<DataVector>*> packaged_gamma2,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_interface_unit_normal,
      gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
          packaged_char_speeds,

      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const Scalar<DataVector>& psi,

      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
      const Scalar<DataVector>& constraint_gamma1,
      const Scalar<DataVector>& constraint_gamma2,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,

      const Scalar<DataVector>& v_psi_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_int,
      const Scalar<DataVector>& v_plus_int,
      const Scalar<DataVector>& v_minus_int,
      const Scalar<DataVector>& gamma2_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_int,
      const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_int,

      const Scalar<DataVector>& v_psi_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_ext,
      const Scalar<DataVector>& v_plus_ext,
      const Scalar<DataVector>& v_minus_ext,
      const Scalar<DataVector>& gamma2_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_ext,
      const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext,
      dg::Formulation /*dg_formulation*/) const noexcept;
};
}  // namespace CurvedScalarWave::BoundaryCorrections
