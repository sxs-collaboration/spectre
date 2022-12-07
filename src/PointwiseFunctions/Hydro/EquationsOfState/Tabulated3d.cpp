// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace EquationsOfState {

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated3D<IsRelativistic>, double, 3)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated3D<IsRelativistic>, DataVector, 3)

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> Tabulated3D<IsRelativistic>::
    equilibrium_electron_fraction_from_density_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature) const {
  Scalar<DataType> electron_fraction = make_with_value<Scalar<DataType>>(
      rest_mass_density, electron_fraction_lower_bound());

  Scalar<DataType> converted_electron_fraction;
  Scalar<DataType> log_rest_mass_density;
  Scalar<DataType> log_temperature;

  convert_to_table_quantities(
      make_not_null(&converted_electron_fraction),
      make_not_null(&log_rest_mass_density), make_not_null(&log_temperature),
      electron_fraction, rest_mass_density, temperature);

  // Compute free-streaming beta-eq. electron fraction (from DeltaMu==0)

  if constexpr (std::is_same_v<DataType, double>) {
    const auto& log_rho = get(log_rest_mass_density);
    const auto& log_T = get(log_temperature);

    const auto f = [this, log_rho, log_T](const double ye) {

      const auto weights = interpolator_.get_weights(log_T, log_rho, ye);
      const auto interpolated_values =
          interpolator_.template interpolate<DeltaMu>(weights);

      return interpolated_values[0];
    };

    const auto root_from_lambda =
        RootFinder::toms748(f, electron_fraction_lower_bound(),
                            electron_fraction_upper_bound(), 1.0e-14, 1.0e-15);

    get(converted_electron_fraction) = root_from_lambda;

  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    for (size_t s = 0; s < electron_fraction.size(); ++s) {
      const auto& log_rho = get(log_rest_mass_density)[s];
      const auto& log_T = get(log_temperature)[s];

      const auto f = [this, log_rho, log_T](const double ye) {

        const auto weights = interpolator_.get_weights(log_T, log_rho, ye);
        const auto interpolated_values =
            interpolator_.template interpolate<DeltaMu>(weights);

        return interpolated_values[0];
      };

      const auto root_from_lambda = RootFinder::toms748(
          f, electron_fraction_lower_bound(), electron_fraction_upper_bound(),
          1.0e-14, 1.0e-15);

      get(converted_electron_fraction)[s] = root_from_lambda;
    }
  }
  return converted_electron_fraction;
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 3>>
Tabulated3D<IsRelativistic>::get_clone() const {
  auto clone = std::make_unique<Tabulated3D<IsRelativistic>>(*this);
  return std::unique_ptr<EquationOfState<IsRelativistic, 3>>(std::move(clone));
}

template <bool IsRelativistic>
void Tabulated3D<IsRelativistic>::initialize(const h5::EosTable& spectre_eos) {
  // STEP 0: Allocate intermediate data structures for initialization

  auto setup_index_variable = [&spectre_eos](std::string name) {
    auto& available_names = spectre_eos.independent_variable_names();
    size_t i = 0;
    for (i = 0; i < available_names.size(); ++i) {
      if (name == available_names[i]) {
        break;
      }
    }

    auto bounds = spectre_eos.independent_variable_bounds()[i];
    if (spectre_eos.independent_variable_uses_log_spacing()[i]) {
      for (auto& b : bounds) {
        b = std::log(b);
      }
    }
    size_t num_points = spectre_eos.independent_variable_number_of_points()[i];

    std::vector<double> index_variable(num_points);

    for (i = 0; i < num_points; ++i) {
      index_variable[i] =
          bounds[0] + (bounds[1] - bounds[0]) / (double(num_points - 1)) * i;
    }

    return index_variable;
  };

  std::vector<double> electron_fraction =
      setup_index_variable("electron fraction");
  std::vector<double> log_density = setup_index_variable("number density");
  std::vector<double> log_temperature = setup_index_variable("temperature");

  // Get size of table
  size_t size =
      electron_fraction.size() * log_density.size() * log_temperature.size();

  std::vector<double> table_data(size * NumberOfVars);

  // Need to setup index variables

  auto pressure = spectre_eos.read_quantity("pressure");
  auto eps = spectre_eos.read_quantity("specific internal energy");
  auto cs2 = spectre_eos.read_quantity("sound speed squared");

  auto mu_l = spectre_eos.read_quantity("lepton chemical potential");
  //  WILL BE NEEDED FOR FUTURE PR
  //  auto mu_q = spectre_eos.read_quantity("charge chemical potential");
  //  auto mu_b = spectre_eos.read_quantity("baryon chemical potential");



  double enthalpy_minimum = 1.e99;
  double eps_min = 1.e99;

  for (size_t s = 0; s < size; ++s) {
    eps_min = std::min(eps_min, eps[s]);
  }

  double energy_shift = (eps_min < 0) ? 2. * eps_min : 0.;

  // STEP 3: Fun with indices

  // nb is in units of 1/fm^3
  // convert to geometrical units
  // assuming an effective mass scale
  // set by the neutron mass

  constexpr double nb_fm3_to_geom = 0.002711492496730566;

  for (size_t s = 0; s < log_density.size(); ++s) {
    log_density[s] += std::log(nb_fm3_to_geom);
  }

  // Convert table
  for (size_t iR = 0; iR < log_density.size(); ++iR) {
    for (size_t iT = 0; iT < log_temperature.size(); ++iT) {
      for (size_t iY = 0; iY < electron_fraction.size(); ++iY) {
        // Index spectre table
       // Ye varies fastest
        size_t index_spectre =
           iY + electron_fraction.size() * (iR + log_density.size() * iT);
        // Local index
        // T varies fastest
        size_t index_tab3D =
            iT + log_temperature.size() * (iR + log_density.size() * iY);

        constexpr double press_MeV_to_geom = 2.885900818968523e-06;

        double* table_point = &(table_data[index_tab3D * NumberOfVars]);

        table_point[Pressure] =
            std::log(press_MeV_to_geom * pressure[index_spectre]);
        table_point[Epsilon] = std::log(eps[index_spectre] - energy_shift);
        table_point[CsSquared] = cs2[index_spectre];
        table_point[DeltaMu] = mu_l[index_spectre];

        // Determine specific enthalpy minimum
        double h = 1. + table_point[Epsilon] +
                 table_point[Pressure] / std::exp(log_density[iR]);
        enthalpy_minimum = std::min(enthalpy_minimum, h);
      }
    }
  }

  initialize(electron_fraction, log_density, log_temperature, table_data,
             energy_shift, enthalpy_minimum);
}

template <bool IsRelativistic>
void Tabulated3D<IsRelativistic>::initialize(
    std::vector<double> electron_fraction, std::vector<double> log_density,
    std::vector<double> log_temperature, std::vector<double> table_data,
    double energy_shift, double enthalpy_minimum) {
  energy_shift_ = energy_shift;
  enthalpy_minimum_ = enthalpy_minimum;
  table_electron_fraction_ = std::move(electron_fraction);
  table_log_density_ = std::move(log_density);
  table_log_temperature_ = std::move(log_temperature);
  table_data_ = std::move(table_data);
  // Need to table

  Index<3> num_x_points;

  // The order is T, rho, Ye
  num_x_points[0] = table_log_temperature_.size();
  num_x_points[1] = table_log_density_.size();
  num_x_points[2] = table_electron_fraction_.size();

  std::array<gsl::span<double const>, 3> independent_data_view;

  independent_data_view[0] =
      gsl::span<double const>{table_log_temperature_.data(), num_x_points[0]};

  independent_data_view[1] =
      gsl::span<double const>{table_log_density_.data(), num_x_points[1]};

  independent_data_view[2] =
      gsl::span<double const>{table_electron_fraction_.data(), num_x_points[2]};

  interpolator_ = intrp::UniformMultiLinearSpanInterpolation<3, NumberOfVars>(
      independent_data_view, {table_data_.data(), table_data_.size()},
      num_x_points);
}

template <bool IsRelativistic>
bool Tabulated3D<IsRelativistic>::is_equal(
    const EquationOfState<IsRelativistic, 3>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Tabulated3D<IsRelativistic>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <bool IsRelativistic>
bool Tabulated3D<IsRelativistic>::operator==(
    const Tabulated3D<IsRelativistic>& rhs) const {
  bool result = true;
  result &= (rhs.enthalpy_minimum_ == this->enthalpy_minimum_);
  result &= (rhs.energy_shift_ == this->energy_shift_);
  result &= (rhs.table_electron_fraction_ == this->table_electron_fraction_);
  result &= (rhs.table_log_density_ == this->table_log_density_);
  result &= (rhs.table_log_temperature_ == this->table_log_temperature_);
  result &= (rhs.table_data_ == this->table_data_);

  return result;
}

template <bool IsRelativistic>
bool Tabulated3D<IsRelativistic>::operator!=(
    const Tabulated3D<IsRelativistic>& rhs) const {
  return not(*this == rhs);
}

template <bool IsRelativistic>
Tabulated3D<IsRelativistic>::Tabulated3D(CkMigrateMessage* msg)
    : EquationOfState<IsRelativistic, 3>(msg) {}
template <bool IsRelativistic>
template <class DataType>
void Tabulated3D<IsRelativistic>::enforce_physicality(
    Scalar<DataType>& electron_fraction, Scalar<DataType>& rest_mass_density,
    Scalar<DataType>& temperature) const {
  if constexpr (std::is_same_v<DataType, double>) {
    get(rest_mass_density) = std::max(
        std::min(get(rest_mass_density), rest_mass_density_upper_bound()),
        rest_mass_density_lower_bound());

    get(electron_fraction) = std::max(
        std::min(get(electron_fraction), electron_fraction_upper_bound()),
        electron_fraction_lower_bound());

    get(temperature) =
        std::max(std::min(get(temperature), temperature_upper_bound()),
                 temperature_lower_bound());

  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    for (size_t s = 0; s < electron_fraction.size(); ++s) {
      get(rest_mass_density)[s] = std::max(
          std::min(get(rest_mass_density)[s], rest_mass_density_upper_bound()),
          rest_mass_density_lower_bound());

      get(electron_fraction)[s] = std::max(
          std::min(get(electron_fraction)[s], electron_fraction_upper_bound()),
          electron_fraction_lower_bound());

      get(temperature)[s] =
          std::max(std::min(get(temperature)[s], temperature_upper_bound()),
                   temperature_lower_bound());
    }
  }
}


template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
Tabulated3D<IsRelativistic>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& electron_fraction) const {
  auto temperature = temperature_from_density_and_energy_impl(
      rest_mass_density, specific_internal_energy, electron_fraction);

  return pressure_from_density_and_temperature_impl(
      rest_mass_density, temperature, electron_fraction);
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
Tabulated3D<IsRelativistic>::pressure_from_density_and_temperature_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& electron_fraction) const {
  Scalar<DataType> converted_electron_fraction;
  Scalar<DataType> log_rest_mass_density;
  Scalar<DataType> log_temperature;

  convert_to_table_quantities(
      make_not_null(&converted_electron_fraction),
      make_not_null(&log_rest_mass_density), make_not_null(&log_temperature),
      electron_fraction, rest_mass_density, temperature);

  Scalar<DataType> pressure =
      make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);

  if constexpr (std::is_same_v<DataType, double>) {
    auto weights = interpolator_.get_weights(get(log_temperature),
                                             get(log_rest_mass_density),
                                             get(converted_electron_fraction));
    auto interpolated_state =
        interpolator_.template interpolate<Pressure>(weights);
    get(pressure) = std::exp(interpolated_state[0]);

  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    for (size_t s = 0; s < electron_fraction.size(); ++s) {
      auto weights = interpolator_.get_weights(
          get(log_temperature)[s], get(log_rest_mass_density)[s],
          get(converted_electron_fraction)[s]);
      auto interpolated_state =
          interpolator_.template interpolate<Pressure>(weights);
      get(pressure)[s] = std::exp(interpolated_state[0]);
    }
  }

  return pressure;
}


template <bool IsRelativistic>
void Tabulated3D<IsRelativistic>::pup(PUP::er& p) {
  EquationOfState<IsRelativistic, 3>::pup(p);
  p | energy_shift_;
  p | enthalpy_minimum_;
  p | table_electron_fraction_;
  p | table_log_density_;
  p | table_log_temperature_;
  p | table_data_;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
Tabulated3D<IsRelativistic>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& electron_fraction) const {
  Scalar<DataType> converted_electron_fraction;
  Scalar<DataType> log_rest_mass_density;

  Scalar<DataType> log_temperature;
  Scalar<DataType> temperature;

  temperature = make_with_value<Scalar<DataType>>(rest_mass_density,
                                                  temperature_lower_bound());

  convert_to_table_quantities(
      make_not_null(&converted_electron_fraction),
      make_not_null(&log_rest_mass_density), make_not_null(&log_temperature),
      electron_fraction, rest_mass_density, temperature);

  // Check bounds on eps, note that eps may be negative
  Scalar<DataType> log_specific_internal_energy = specific_internal_energy;

  if constexpr (std::is_same_v<DataType, double>) {
    get(log_specific_internal_energy) =
        std::max(std::min(get(specific_internal_energy),
                          specific_internal_energy_upper_bound(
                              get(rest_mass_density), get(electron_fraction))),
                 specific_internal_energy_lower_bound(get(rest_mass_density),
                                                      get(electron_fraction)));
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    for (size_t s = 0; s < electron_fraction.size(); ++s) {
      get(log_specific_internal_energy)[s] = std::max(
          std::min(get(specific_internal_energy)[s],
                   specific_internal_energy_upper_bound(
                       get(rest_mass_density)[s], get(electron_fraction)[s])),
          specific_internal_energy_lower_bound(get(rest_mass_density)[s],
                                               get(electron_fraction)[s]));
    }
  }

  // Correct for negative eps
  get(log_specific_internal_energy) -= energy_shift_;
  get(log_specific_internal_energy) = log(get(log_specific_internal_energy));

  if constexpr (std::is_same_v<DataType, double>) {
    const auto& log_eps = get(log_specific_internal_energy);
    const auto& log_rho = get(log_rest_mass_density);
    const auto& ye = get(converted_electron_fraction);

    // Root-finding appropriate between reference density and maximum density
    // We can use x=0 and x=x_max as bounds
    const auto f = [this, log_eps, log_rho, ye](const double log_T) {

      const auto weights = interpolator_.get_weights(log_T, log_rho, ye);
      const auto interpolated_values =
          interpolator_.template interpolate<Epsilon>(weights);

      return log_eps - interpolated_values[0];
    };

    const auto root_from_lambda = RootFinder::toms748(
        f, table_log_temperature_.front(),
        upper_bound_tolerance_ * table_log_temperature_.back(), 1.0e-14,
        1.0e-15);

    get(temperature) = exp(root_from_lambda);

  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    for (size_t s = 0; s < electron_fraction.size(); ++s) {
      const auto& log_eps = get(log_specific_internal_energy)[s];
      const auto& log_rho = get(log_rest_mass_density)[s];
      const auto& ye = get(converted_electron_fraction)[s];

      // Root-finding appropriate between reference density and maximum density
      // We can use x=0 and x=x_max as bounds
      const auto f = [this, log_eps, log_rho, ye](const double log_T) {

        const auto weights = interpolator_.get_weights(log_T, log_rho, ye);
        const auto interpolated_values =
            interpolator_.template interpolate<Epsilon>(weights);

        return log_eps - interpolated_values[0];
      };
      const auto root_from_lambda = RootFinder::toms748(
          f, table_log_temperature_.front(),
          upper_bound_tolerance_ * table_log_temperature_.back(), 1.0e-14,
          1.0e-15);

      get(temperature)[s] = exp(root_from_lambda);
    }
  }
  return temperature;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> Tabulated3D<IsRelativistic>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature,
        const Scalar<DataType>& electron_fraction) const {
  Scalar<DataType> converted_electron_fraction;
  Scalar<DataType> log_rest_mass_density;
  Scalar<DataType> log_temperature;

  convert_to_table_quantities(
      make_not_null(&converted_electron_fraction),
      make_not_null(&log_rest_mass_density), make_not_null(&log_temperature),
      electron_fraction, rest_mass_density, temperature);

  Scalar<DataType> specific_internal_energy =
      make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);

  if constexpr (std::is_same_v<DataType, double>) {
    auto weights = interpolator_.get_weights(get(log_temperature),
                                             get(log_rest_mass_density),
                                             get(converted_electron_fraction));
    auto interpolated_state =
        interpolator_.template interpolate<Epsilon>(weights);
    get(specific_internal_energy) =
        std::exp(interpolated_state[0]) + energy_shift_;
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    for (size_t s = 0; s < electron_fraction.size(); ++s) {
      auto weights = interpolator_.get_weights(
          get(log_temperature)[s], get(log_rest_mass_density)[s],
          get(converted_electron_fraction)[s]);
      auto interpolated_state =
          interpolator_.template interpolate<Epsilon>(weights);
      get(specific_internal_energy)[s] =
          std::exp(interpolated_state[0]) + energy_shift_;
    }
  }

  return specific_internal_energy;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> Tabulated3D<IsRelativistic>::
    sound_speed_squared_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature,
        const Scalar<DataType>& electron_fraction) const {
  Scalar<DataType> converted_electron_fraction;
  Scalar<DataType> log_rest_mass_density;
  Scalar<DataType> log_temperature;

  convert_to_table_quantities(
      make_not_null(&converted_electron_fraction),
      make_not_null(&log_rest_mass_density), make_not_null(&log_temperature),
      electron_fraction, rest_mass_density, temperature);

  Scalar<DataType> cs2 =
      make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);

  if constexpr (std::is_same_v<DataType, double>) {
    auto weights = interpolator_.get_weights(get(log_temperature),
                                             get(log_rest_mass_density),
                                             get(converted_electron_fraction));
    auto interpolated_state =
        interpolator_.template interpolate<CsSquared>(weights);
    get(cs2) = interpolated_state[0];

  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    for (size_t s = 0; s < electron_fraction.size(); ++s) {
      auto weights = interpolator_.get_weights(
          get(log_temperature)[s], get(log_rest_mass_density)[s],
          get(converted_electron_fraction)[s]);
      auto interpolated_state =
          interpolator_.template interpolate<CsSquared>(weights);
      get(cs2)[s] = interpolated_state[0];
    }
  }

  return cs2;
}

template <bool IsRelativistic>
double Tabulated3D<IsRelativistic>::specific_internal_energy_lower_bound(
    const double rest_mass_density, const double electron_fraction) const {
  double converted_electron_fraction =
      std::min(std::max(electron_fraction_lower_bound(), electron_fraction),
               electron_fraction_upper_bound());

  double log_rest_mass_density =
      std::min(std::max(rest_mass_density_lower_bound(), rest_mass_density),
               rest_mass_density_upper_bound());

  log_rest_mass_density = log(log_rest_mass_density);

  auto weights = interpolator_.get_weights(log(temperature_lower_bound()),
                                           log_rest_mass_density,
                                           converted_electron_fraction);
  auto interpolated_state =
      interpolator_.template interpolate<Epsilon>(weights);

  return exp(interpolated_state[0]) + energy_shift_;
}

template <bool IsRelativistic>
double Tabulated3D<IsRelativistic>::specific_internal_energy_upper_bound(
    const double rest_mass_density, const double electron_fraction) const {
  double converted_electron_fraction =
      std::min(std::max(electron_fraction_lower_bound(), electron_fraction),
               electron_fraction_upper_bound());

  double log_rest_mass_density =
      std::min(std::max(rest_mass_density_lower_bound(), rest_mass_density),
               rest_mass_density_upper_bound());

  log_rest_mass_density = log(log_rest_mass_density);

  auto weights = interpolator_.get_weights(
      log(upper_bound_tolerance_ * temperature_upper_bound()),
      log_rest_mass_density, converted_electron_fraction);
  auto interpolated_state =
      interpolator_.template interpolate<Epsilon>(weights);

  return exp(interpolated_state[0]) + energy_shift_;
}

template <bool IsRelativistic>
Tabulated3D<IsRelativistic>::
    Tabulated3D(  // NOLINTNEXTLINE(performance-unnecessary-value-param)
        std::vector<double> electron_fraction,
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
        std::vector<double> log_density,
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
        std::vector<double> log_temperature,
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
        std::vector<double> table_data, double energy_shift,
        double enthalpy_minimum) {
  initialize(std::move(electron_fraction), std::move(log_density),
             std::move(log_temperature), std::move(table_data), energy_shift,
             enthalpy_minimum);
}

template <bool IsRelativistic>
Tabulated3D<IsRelativistic>::Tabulated3D(const h5::EosTable& spectre_eos) {
  initialize(spectre_eos);
}

template <bool IsRelativistic>
Tabulated3D<IsRelativistic>::Tabulated3D(const std::string& filename,
                                         const std::string& subfilename) {
  h5::H5File<h5::AccessType::ReadOnly> eos_file{filename};
  const auto& spectre_eos = eos_file.get<h5::EosTable>("/" + subfilename);

  initialize(spectre_eos);
}

}  // namespace EquationsOfState

template class EquationsOfState::Tabulated3D<true>;
template class EquationsOfState::Tabulated3D<false>;

