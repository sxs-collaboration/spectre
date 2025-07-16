// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/TagsDeclarations.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

/// \brief Utilities for initializing the curved scalar wave system.
namespace CurvedScalarWave::Initialization {
/// \ingroup InitializationGroup
/// \brief Mutator meant to be used with
/// `Initialization::Actions::AddSimpleTags` to initialize the constraint
/// damping parameters of the CurvedScalarWave system
///
/// DataBox changes:
/// - Adds:
///   * `CurvedScalarWave::Tags::ConstraintGamma1`
///   * `CurvedScalarWave::Tags::ConstraintGamma2`
/// - Removes: nothing
/// - Modifies: nothing

template <size_t Dim>
struct InitializeConstraintDampingGammas
    : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags =
      tmpl::list<Tags::ConstraintGamma1, Tags::ConstraintGamma2>;
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> gamma1,
                    const gsl::not_null<Scalar<DataVector>*> gamma2,
                    const Mesh<Dim>& mesh) {
    const size_t number_of_grid_points = mesh.number_of_grid_points();
    *gamma1 = Scalar<DataVector>{number_of_grid_points, 0.};
    *gamma2 = Scalar<DataVector>{number_of_grid_points, 1.};
  }
};

/*!
 * \brief Analytic initial data for scalar waves in curved spacetime
 *
 * \details When evolving a scalar field propagating through curved spacetime,
 * this mutator initializes the scalar field and spacetime variables using
 *
 * 1. analytic solution(s) or data of the flat or curved scalar wave equation
 * for the evolution variables
 * 2. solutions of the Einstein equations for the spacetime background.
 *
 * If the scalar field initial data returns `CurvedScalarWave` tags, \f$\Psi\f$,
 * \f$\Pi\f$ and \f$\Phi_i\f$ will simply be forwarded from the initial data
 * class. Alternatively, the scalar field initial data can be provided using any
 * member class of `ScalarWave::Solutions` which return `ScalarWave` tags. In
 * this case, \f$\Phi_i\f$ and \f$\Psi\f$ will also be forwarded but
 * \f$\Pi\f$ will be adjusted to account for the curved background. Its
 * definition comes from requiring it to be the future-directed time derivative
 * of the scalar field in curved spacetime:
 *
 * \f{align}
 * \Pi :=& -n^a \partial_a \Psi \\
 *     =&  \frac{1}{\alpha}\left(\beta^k \Phi_k - {\partial_t\Psi}\right),\\
 *     =&  \frac{1}{\alpha}\left(\beta^k \Phi_k + {\Pi}_{\mathrm{flat}}\right),
 * \f}
 *
 * where \f$n^a\f$ is the unit normal to spatial slices of the spacetime
 * foliation, and \f${\Pi}_{\mathrm{flat}}\f$ comes from the flat spacetime
 * solution.
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: Tags::Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
 * CurvedScalarWave::Tags::Pi, CurvedScalarWave::Tags::Phi<Dim>>>
 */
template <size_t Dim, typename DerivedClasses>
struct InitializeEvolvedVariables : tt::ConformsTo<db::protocols::Mutator> {
  using flat_variables_tag = typename ScalarWave::System<Dim>::variables_tag;
  using curved_variables_tag =
      typename CurvedScalarWave::System<Dim>::variables_tag;

  using return_tags = tmpl::list<curved_variables_tag>;
  using argument_tags =
      tmpl::list<::Tags::Time, domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 evolution::initial_data::Tags::InitialData,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim>>;

  static void apply(
      const gsl::not_null<typename curved_variables_tag::type*> evolved_vars,
      const double initial_time,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const evolution::initial_data::InitialData& solution_or_data,
      [[maybe_unused]] const Scalar<DataVector>& lapse,
      [[maybe_unused]] const tnsr::I<DataVector, Dim>& shift) {
    call_with_dynamic_type<void, DerivedClasses>(
        &solution_or_data,
        [&evolved_vars, initial_time, &inertial_coords, &lapse,
         &shift]<typename InitialDataSubclass>(
            const InitialDataSubclass* const data_or_solution) {
          if constexpr (is_analytic_data_v<InitialDataSubclass> or
                        is_analytic_solution_v<InitialDataSubclass>) {
            if constexpr (tmpl::list_contains_v<
                              typename InitialDataSubclass::tags,
                              CurvedScalarWave::Tags::Psi>) {
              (void)lapse, (void)shift;
              evolved_vars->assign_subset(
                  evolution::Initialization::initial_data(
                      *data_or_solution, inertial_coords, initial_time,
                      typename curved_variables_tag::tags_list{}));
            } else {
              static_assert(
                  tmpl::list_contains_v<typename InitialDataSubclass::tags,
                                        ScalarWave::Tags::Psi>,
                  "The initial data class must either calculate ScalarWave "
                  "or CurvedScalarWave variables.");
              const auto initial_data = evolution::Initialization::initial_data(
                  *data_or_solution, inertial_coords, initial_time,
                  typename flat_variables_tag::tags_list{});

              get<CurvedScalarWave::Tags::Psi>(*evolved_vars) =
                  get<ScalarWave::Tags::Psi>(initial_data);
              get<CurvedScalarWave::Tags::Phi<Dim>>(*evolved_vars) =
                  get<ScalarWave::Tags::Phi<Dim>>(initial_data);
              const auto shift_dot_dpsi = dot_product(
                  shift, get<ScalarWave::Tags::Phi<Dim>>(initial_data));
              get(get<CurvedScalarWave::Tags::Pi>(*evolved_vars)) =
                  (get(shift_dot_dpsi) +
                   get(get<ScalarWave::Tags::Pi>(initial_data))) /
                  get(lapse);
            }
          } else {
            ERROR(
                "Trying to use "
                "'evolution::Initialization::Actions::SetVariables' with a "
                "class that's not marked as analytic solution or analytic "
                "data. To support numeric initial data, add a "
                "system-specific initialization routine to your executable.");
          }
        });
  }
};
}  // namespace CurvedScalarWave::Initialization
