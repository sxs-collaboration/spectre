// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Particles::MonteCarlo {

/// Class responsible for reading neutrino-matter interaction
/// tables.
template <size_t EnergyBins, size_t NeutrinoSpecies>
class NeutrinoInteractionTable : public PUP::able {
 public:
  /// Read table from disk and stores interaction rates.
  explicit NeutrinoInteractionTable(const std::string& filename);

  /// Explicit instantiation from table values, for tests
  NeutrinoInteractionTable(
      std::vector<double> table_data_,
      const std::array<double, EnergyBins>& table_neutrino_energies_,
      std::vector<double> table_log_density_,
      std::vector<double> table_log_temperature_,
      std::vector<double> table_electron_fraction_);

  explicit NeutrinoInteractionTable(CkMigrateMessage* msg) : PUP::able(msg) {}

  using PUP::able::register_constructor;
  void pup(PUP::er& p) override;
  WRAPPED_PUPable_decl_template(NeutrinoInteractionTable);

  /// Interpolate interaction rates to given values of density,
  /// temperature and electron fraction.
  void get_neutrino_matter_interactions(
      gsl::not_null<
          std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
          emissivity_in_cell,
      gsl::not_null<
          std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
          absorption_opacity,
      gsl::not_null<
          std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
          scattering_opacity,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& temperature,
      const double& minimum_temperature) const;

  const std::array<double, EnergyBins>& get_neutrino_energies() const {
    return table_neutrino_energies;
  }

 private:
  void initialize_interpolator();

  // Stores emissivity, absorption_opacity, scattering_opacity
  // For each quantities, there are NeutrinoSpecies * EnergyBins
  // variables store as a function of log(density), log(temperature)
  // and electron fraction.
  // The indexing varies fastest in EnergyBins, then Species, then
  // log(density), then log(temperature), and finally Ye.
  std::vector<double> table_data{};
  // Central energy of each bin
  std::array<double, EnergyBins> table_neutrino_energies;
  // Table discretization
  std::vector<double> table_log_density{};
  std::vector<double> table_log_temperature{};
  std::vector<double> table_electron_fraction{};

  intrp::UniformMultiLinearSpanInterpolation<3,
                                             3 * EnergyBins * NeutrinoSpecies>
      interpolator_{};

  const double min_kappa = 1.e-70;
  const double max_kappa = 1.e70;
};

}  // namespace Particles::MonteCarlo
