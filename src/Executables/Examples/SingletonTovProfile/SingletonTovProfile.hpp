// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>
#include <pup.h>

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/EquationsOfState/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"

/*!
 * \page SingletonTovProfileExecutablePage SingletonTovProfile Executable
 * \tableofcontents
 * The %SingletonTovProfile executable solves the TOV equations for the
 * provided equation of state with the parameters given in the
 * `tests/InputFiles/ExampleExecutables/SingletonTovProfile.yaml` input file.
 *
 * The results of the solution are then written to the output file
 * specified in the yaml file. A summary of the results are also printed
 * to the terminal.
 */

namespace TovSingletonCacheTags {

struct Name {
  using type = std::string;
  static constexpr OptionString help{"The name of the output file"};
};

struct CentralMassDensity {
  using type = double;
  static constexpr OptionString help{"The central mass density rho_c"};
};

/*!
 *  The points are linearly spaced in the areal radius. For
 *  reconstruction purposes, the barycentric rational interpolant to the
 *  log_enthalpy-radius curve is the most well-behaved, and
 *  is best interpolated using an order 3 barycentric rational interpolant.
 *  Using an interpolant with 25 points results in a log enthalpy interpolant
 *  that has a relative error of at most 1.e-8 at all radii, relative to the
 *  numerically found result.
 */
struct NumberOfPoints {
  using type = size_t;
  static constexpr OptionString help{"The number of data points to output"};
};
}  // namespace TovSingletonCacheTags

namespace Actions {
struct SolveTovProfile {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const std::string& filename =
        Parallel::get<TovSingletonCacheTags::Name>(cache);
    const double central_mass_density =
        Parallel::get<TovSingletonCacheTags::CentralMassDensity>(cache);
    const size_t num_pts =
        Parallel::get<TovSingletonCacheTags::NumberOfPoints>(cache);
    const auto& equation_of_state =
        Parallel::get<OptionTags::EquationOfState<true, 1>>(cache);

    // Integrate TOV equations out to the surface of the star:
    const double surface_log_enthalpy = 0.0;
    const gr::Solutions::TovSolution tov_star(
        equation_of_state, central_mass_density, surface_log_enthalpy);
    const double outer_radius = tov_star.outer_radius();

    // Define helper closures:
    const auto pressure_from_density = [&equation_of_state](
        const double density) noexcept {
      return equation_of_state.pressure_from_density(Scalar<double>{density})
          .get();
    };
    const auto energy_from_density = [&equation_of_state](
        const double density) noexcept {
      return equation_of_state
          .specific_internal_energy_from_density(Scalar<double>{density})
          .get();
    };
    const auto sound_speed_squared_from_density = [&equation_of_state](
        const double density) noexcept {
      return equation_of_state.chi_from_density(Scalar<double>{density}).get() +
             equation_of_state
                 .kappa_times_p_over_rho_squared_from_density(
                     Scalar<double>{density})
                 .get();
    };

    // Detailed output is written to the H5 file:
    const std::string h5_file_name = filename + ".h5";
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
    h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
    std::vector<std::string> legend{
        "Radius", "Mass",         "Pressure",           "Density",
        "Energy", "Log Enthalpy", "Sound speed squared"};
    const uint32_t version_number = 0;
    auto& tov_dat_file =
        my_file.insert<h5::Dat>("/TovProfile", legend, version_number);
    const double delta_r_finer = outer_radius / (num_pts - 1.0);
    for (size_t i = 0; i < num_pts; i++) {
      const auto current_radius = i * delta_r_finer;
      const auto enclosed_mass = tov_star.mass(current_radius);
      const auto log_enthalpy = tov_star.log_specific_enthalpy(current_radius);
      const auto enthalpy = std::exp(log_enthalpy);
      const auto density = (equation_of_state.rest_mass_density_from_enthalpy(
                                Scalar<double>{enthalpy}))
                               .get();
      const auto pressure = pressure_from_density(density);
      const auto energy = energy_from_density(density);
      const auto sound_speed_squared =
          sound_speed_squared_from_density(density);
      std::vector<double> row_of_tov_data{
          current_radius, enclosed_mass, pressure,           density,
          energy,         log_enthalpy,  sound_speed_squared};
      tov_dat_file.append(row_of_tov_data);
    }
    my_file.close_current_object();
  }
};
}  // namespace Actions

template <class Metavariables>
struct SingletonTovProfile {
  using const_global_cache_tag_list =
      tmpl::list<TovSingletonCacheTags::Name,
                 TovSingletonCacheTags::CentralMassDensity,
                 TovSingletonCacheTags::NumberOfPoints,
                 OptionTags::EquationOfState<true, 1>>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
  using options = tmpl::list<>;
  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>&) noexcept {}
  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept;
};

template <class Metavariables>
void SingletonTovProfile<Metavariables>::execute_next_phase(
    const typename Metavariables::Phase /* next_phase */,
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
  Parallel::simple_action<Actions::SolveTovProfile>(
      Parallel::get_parallel_component<SingletonTovProfile>(
          *(global_cache.ckLocalBranch())));
}

struct Metavars {
  using const_global_cache_tag_list = tmpl::list<>;

  using component_list = tmpl::list<SingletonTovProfile<Metavars>>;

  static constexpr OptionString help{
      "Write the profile of a single TOV star to an h5 file."};

  enum class Phase { Initialization, Execute, Exit };

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        Metavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Execute
                                                  : Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &Parallel::register_derived_classes_with_charm<
                               EquationsOfState::EquationOfState<1, true>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
