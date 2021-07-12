// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace GeneralizedHarmonic::BoundaryConditions::detail {
enum class ConstraintPreservingBjorhusType {
  ConstraintPreserving,
  ConstraintPreservingPhysical
};

ConstraintPreservingBjorhusType
convert_constraint_preserving_bjorhus_type_from_yaml(
    const Options::Option& options);
}  // namespace GeneralizedHarmonic::BoundaryConditions::detail

namespace GeneralizedHarmonic::BoundaryConditions {
/*!
 * \brief Sets constraint preserving boundary conditions using the Bjorhus
 * method.
 *
 * \details Boundary conditions for the generalized harmonic evolution system
 * can be divided in to three parts, constraint-preserving, physical and gauge
 * boundary conditions.
 *
 * The generalized harmonic (GH) evolution system is a first-order reduction of
 * Einstein equations brought about by the imposition of GH gauge. This
 * introduces constraints on the free (evolved) variables in addition to the
 * standard Hamiltonian and momentum constraints. The constraint-preserving
 * portion of the boundary conditions is designed to prevent the influx of
 * constraint violations from external faces of the evolution domain, by damping
 * them away on a controlled and short time-scale. These conditions are imposed
 * as corrections to the projections of the right-hand-sides of the GH evolution
 * equations (i.e. using Bjorhus' method \cite Bjorhus1995), and are
 * written down in Eq. (63) - (65) of \cite Lindblom2005qh . The gauge degrees
 * of freedom are controlled by imposing a Sommerfeld-type condition (\f$L=0\f$
 * member of the hierarchy derived in \cite BaylissTurkel) that allow gauge
 * perturbations to pass through the boundary without strong reflections. These
 * assume a spherical outer boundary, and can be written down as in Eq. (25) of
 * \cite Rinne2007ui . Finally, the physical boundary conditions control the
 * influx of inward propagating gravitational-wave solutions from the external
 * boundaries. These are derived by considering the evolution system of the Weyl
 * curvature tensor, and controlling the inward propagating characteristics of
 * the system that are proportional to the Newman-Penrose curvature spinor
 * components \f$\Psi_4\f$ and \f$\Psi_0\f$. Here we use Eq. (68) of
 * \cite Lindblom2005qh to disallow any incoming waves. It is to be noted that
 * all the above conditions are also imposed on characteristic modes with speeds
 * exactly zero.
 *
 * This class provides two choices of combinations of the above corrections:
 *  - `ConstraintPreserving` : this imposes the constraint-preserving and
 * gauge-controlling corrections;
 *  - `ConstraintPreservingPhysical` : this additionally restricts the influx of
 * any physical gravitational waves from the outer boundary, in addition to
 * preventing the influx of constraint violations and gauge perturbations.
 *
 * We refer to `Bjorhus::constraint_preserving_bjorhus_corrections_dt_v_psi()`,
 * `Bjorhus::constraint_preserving_bjorhus_corrections_dt_v_zero()`,
 * `Bjorhus::constraint_preserving_bjorhus_corrections_dt_v_minus()`, and
 * `Bjorhus::constraint_preserving_physical_bjorhus_corrections_dt_v_minus()`
 * for the further details on implementation.
 *
 * \note These boundary conditions assume a spherical outer boundary. Also, we
 * do not yet have an option to inject incoming gravitational waves at the outer
 * boundary.
 */
template <size_t Dim>
class ConstraintPreservingBjorhus final : public BoundaryCondition<Dim> {
 public:
  struct TypeOptionTag {
    using type = detail::ConstraintPreservingBjorhusType;
    static std::string name() noexcept { return "Type"; }
    static constexpr Options::String help{
        "Whether to impose ConstraintPreserving, with or without physical "
        "terms for VMinus."};
  };

  using options = tmpl::list<TypeOptionTag>;
  static constexpr Options::String help{
      "ConstraintPreservingBjorhus boundary conditions setting the value of the"
      "time derivatives of the spacetime metric, Phi and Pi to expressions that"
      "prevent the influx of constraint violations and reflections."};
  static std::string name() noexcept { return "ConstraintPreservingBjorhus"; }

  ConstraintPreservingBjorhus(
      detail::ConstraintPreservingBjorhusType type) noexcept;

  ConstraintPreservingBjorhus() = default;
  /// \cond
  ConstraintPreservingBjorhus(ConstraintPreservingBjorhus&&) noexcept = default;
  ConstraintPreservingBjorhus& operator=(
      ConstraintPreservingBjorhus&&) noexcept = default;
  ConstraintPreservingBjorhus(const ConstraintPreservingBjorhus&) = default;
  ConstraintPreservingBjorhus& operator=(const ConstraintPreservingBjorhus&) =
      default;
  /// \endcond
  ~ConstraintPreservingBjorhus() override = default;

  explicit ConstraintPreservingBjorhus(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      ConstraintPreservingBjorhus);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
                 Tags::Pi<Dim, Frame::Inertial>,
                 Tags::Phi<Dim, Frame::Inertial>>;
  using dg_interior_temporary_tags = tmpl::list<
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      ConstraintDamping::Tags::ConstraintGamma1,
      ConstraintDamping::Tags::ConstraintGamma2, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalVector<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial, DataVector>,
      Tags::ThreeIndexConstraint<Dim, Frame::Inertial>,
      Tags::GaugeH<Dim, Frame::Inertial>,
      Tags::SpacetimeDerivGaugeH<Dim, Frame::Inertial>>;
  using dg_interior_dt_vars_tags = tmpl::list<
      ::Tags::dt<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>,
      ::Tags::dt<Tags::Pi<Dim, Frame::Inertial>>,
      ::Tags::dt<Tags::Phi<Dim, Frame::Inertial>>>;
  using dg_interior_deriv_vars_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Pi<Dim, Frame::Inertial>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::deriv<Tags::Phi<Dim, Frame::Inertial>, tmpl::size_t<Dim>,
                    Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          dt_spacetime_metric_correction,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          dt_pi_correction,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
      // c.f. dg_interior_evolved_variables_tags
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      // c.f. dg_interior_temporary_tags
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::AA<DataVector, Dim, Frame::Inertial>&
          inverse_spacetime_metric,
      const tnsr::A<DataVector, Dim, Frame::Inertial>&
          spacetime_unit_normal_vector,
      const tnsr::a<DataVector, Dim, Frame::Inertial>&
          spacetime_unit_normal_one_form,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
      const tnsr::a<DataVector, Dim, Frame::Inertial>& gauge_source,
      const tnsr::ab<DataVector, Dim, Frame::Inertial>&
          spacetime_deriv_gauge_source,
      // c.f. dg_interior_dt_vars_tags
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& dt_phi,
      // c.f. dg_interior_deriv_vars_tags
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_spacetime_metric,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_pi,
      const tnsr::ijaa<DataVector, Dim, Frame::Inertial>& d_phi) const noexcept;

 private:
  void compute_intermediate_vars(
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          unit_interface_normal_vector,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          four_index_constraint,
      gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          inverse_spatial_metric,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          extrinsic_curvature,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          incoming_null_one_form,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          outgoing_null_one_form,
      gsl::not_null<tnsr::A<DataVector, Dim, Frame::Inertial>*>
          incoming_null_vector,
      gsl::not_null<tnsr::A<DataVector, Dim, Frame::Inertial>*>
          outgoing_null_vector,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> projection_ab,
      gsl::not_null<tnsr::Ab<DataVector, Dim, Frame::Inertial>*> projection_Ab,
      gsl::not_null<tnsr::AA<DataVector, Dim, Frame::Inertial>*> projection_AB,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          char_projected_rhs_dt_v_psi,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          char_projected_rhs_dt_v_zero,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          char_projected_rhs_dt_v_minus,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          constraint_char_zero_plus,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          constraint_char_zero_minus,
      gsl::not_null<std::array<DataVector, 4>*> char_speeds,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::AA<DataVector, Dim, Frame::Inertial>&
          inverse_spacetime_metric,
      const tnsr::A<DataVector, Dim, Frame::Inertial>&
          spacetime_unit_normal_vector,
      const tnsr::a<DataVector, Dim, Frame::Inertial>&
          spacetime_unit_normal_one_form,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
      const tnsr::a<DataVector, Dim, Frame::Inertial>& gauge_source,
      const tnsr::ab<DataVector, Dim, Frame::Inertial>&
          spacetime_deriv_gauge_source,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& dt_phi,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_spacetime_metric,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_pi,
      const tnsr::ijaa<DataVector, Dim, Frame::Inertial>& d_phi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_spacetime_metric)
      const noexcept;

  detail::ConstraintPreservingBjorhusType type_{
      detail::ConstraintPreservingBjorhusType::ConstraintPreservingPhysical};
};
}  // namespace GeneralizedHarmonic::BoundaryConditions

template <>
struct Options::create_from_yaml<GeneralizedHarmonic::BoundaryConditions::
                                     detail::ConstraintPreservingBjorhusType> {
  template <typename Metavariables>
  static typename GeneralizedHarmonic::BoundaryConditions::detail::
      ConstraintPreservingBjorhusType
      create(const Options::Option& options) {
    return GeneralizedHarmonic::BoundaryConditions::detail::
        convert_constraint_preserving_bjorhus_type_from_yaml(options);
  }
};
