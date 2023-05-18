// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <type_traits>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Utilities/TMPL.hpp"

template <bool EvolveCcm>
struct CharacteristicExtractDefaults {
  static constexpr bool evolve_ccm = EvolveCcm;
  using evolved_swsh_tag = Cce::Tags::BondiJ;
  using evolved_swsh_dt_tag = Cce::Tags::BondiH;
  using evolved_coordinates_variables_tag = Tags::Variables<
      tmpl::conditional_t<evolve_ccm,
                          tmpl::list<Cce::Tags::CauchyCartesianCoords,
                                     Cce::Tags::PartiallyFlatCartesianCoords,
                                     Cce::Tags::InertialRetardedTime>,
                          tmpl::list<Cce::Tags::CauchyCartesianCoords,
                                     Cce::Tags::InertialRetardedTime>>>;

  struct swsh_vars_selector {
    static std::string name() { return "SwshVars"; }
  };

  struct coord_vars_selector {
    static std::string name() { return "CoordVars"; }
  };

  using cce_boundary_communication_tags =
      Cce::Tags::characteristic_worldtube_boundary_tags<
          Cce::Tags::BoundaryValue>;

  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Cce::Tags::BondiR, Cce::Tags::DuRDividedByR,
                     Cce::Tags::BondiJ, Cce::Tags::Dr<Cce::Tags::BondiJ>,
                     Cce::Tags::BondiBeta, Cce::Tags::BondiQ, Cce::Tags::BondiU,
                     Cce::Tags::BondiW, Cce::Tags::BondiH>,
          tmpl::bind<Cce::Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Cce::Tags::BondiUAtScri, Cce::Tags::PartiallyFlatGaugeC,
      Cce::Tags::PartiallyFlatGaugeD, Cce::Tags::PartiallyFlatGaugeOmega,
      Cce::Tags::Du<Cce::Tags::PartiallyFlatGaugeOmega>,
      tmpl::conditional_t<
          evolve_ccm,
          tmpl::list<
              Cce::Tags::CauchyGaugeC, Cce::Tags::CauchyGaugeD,
              Cce::Tags::CauchyGaugeOmega,
              Spectral::Swsh::Tags::Derivative<Cce::Tags::CauchyGaugeOmega,
                                               Spectral::Swsh::Tags::Eth>>,
          tmpl::list<>>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::PartiallyFlatGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Cce::all_boundary_pre_swsh_derivative_tags_for_scri,
      Cce::all_boundary_swsh_derivative_tags_for_scri>>;

  using scri_values_to_observe =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::Du<Cce::Tags::TimeIntegral<
                     Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>,
                 Cce::Tags::EthInertialRetardedTime>;

  using cce_scri_tags =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>,
                 Cce::Tags::EthInertialRetardedTime>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      Cce::bondi_hypersurface_step_tags,
      tmpl::bind<Cce::integrand_terms_to_compute_for_bondi_variable,
                 tmpl::_1>>>;
  using ccm_matching_tags = tmpl::list<
      Cce::Tags::BondiJCauchyView, Cce::Tags::Psi0Match,
      Cce::Tags::Dy<Cce::Tags::Psi0Match>,
      Cce::Tags::Psi0, Cce::Tags::Dy<Cce::Tags::BondiJCauchyView>,
      Cce::Tags::Dy<Cce::Tags::Dy<Cce::Tags::BondiJCauchyView>>>;

  using cce_integration_independent_tags = tmpl::conditional_t<
      evolve_ccm,
      tmpl::append<Cce::pre_computation_tags, ccm_matching_tags,
                   tmpl::list<Cce::Tags::DuRDividedByR>>,
      tmpl::push_back<Cce::pre_computation_tags, Cce::Tags::DuRDividedByR>>;

  using cce_temporary_equations_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<cce_integrand_tags,
                      tmpl::bind<Cce::integrand_temporary_tags, tmpl::_1>>>>;
  using cce_pre_swsh_derivatives_tags = Cce::all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = Cce::all_transform_buffer_tags;
  using cce_swsh_derivative_tags = Cce::all_swsh_derivative_tags;
  using cce_angular_coordinate_tags =
      tmpl::conditional_t<evolve_ccm,
                          tmpl::list<Cce::Tags::CauchyAngularCoords,
                                     Cce::Tags::PartiallyFlatAngularCoords>,
                          tmpl::list<Cce::Tags::CauchyAngularCoords>>;
  using cce_step_choosers = tmpl::list<
      StepChoosers::Constant<StepChooserUse::LtsStep>,
      StepChoosers::Increase<StepChooserUse::LtsStep>,
      StepChoosers::ErrorControl<StepChooserUse::LtsStep,
                                 Tags::Variables<tmpl::list<evolved_swsh_tag>>,
                                 swsh_vars_selector>,
      StepChoosers::ErrorControl<StepChooserUse::LtsStep,
                                 evolved_coordinates_variables_tag,
                                 coord_vars_selector>>;

  using ccm_psi0 = tmpl::list<
      Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>,
      Cce::Tags::BoundaryValue<Cce::Tags::Dlambda<Cce::Tags::Psi0Match>>>;
};
