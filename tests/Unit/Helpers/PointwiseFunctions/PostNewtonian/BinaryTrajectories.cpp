// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/PointwiseFunctions/PostNewtonian/BinaryTrajectories.hpp"

#include <array>
#include <cmath>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

BinaryTrajectories::BinaryTrajectories(
    double initial_separation,
    const std::array<double, 3>& center_of_mass_velocity, bool newtonian)
    : initial_separation_fourth_power_{square(square(initial_separation))},
      center_of_mass_velocity_(center_of_mass_velocity),
      newtonian_(newtonian) {}

template <typename DataType>
DataType BinaryTrajectories::separation(const DataType& time) const {
  const auto pn_correction_term = (newtonian_ ? 0.0 : 12.8) * time;
  return pow(initial_separation_fourth_power_ - pn_correction_term, 0.25);
}

template <typename DataType>
DataType BinaryTrajectories::orbital_frequency(const DataType& time) const {
  return pow(separation(time), -1.5);
}

template <typename DataType>
DataType BinaryTrajectories::angular_velocity(const DataType& time) const {
  // This is d/dt(orbital_frequency) if we are using PN, but 0 if it's newtonian
  const auto pn_correction_term =
      (newtonian_ ? 0.0 : 4.8) *
      pow(initial_separation_fourth_power_ - 12.8 * time, -1.375);
  return orbital_frequency(time) + pn_correction_term * time;
}

template <typename DataType>
std::array<tnsr::I<DataType, 3>, 2> BinaryTrajectories::positions(
    const DataType& time) const {
  return position_impl(time, separation(time));
}

template <typename DataType>
std::array<tnsr::I<DataType, 3>, 2> BinaryTrajectories::positions_no_expansion(
    const DataType& time) const {
  // Separation stays constant while orbital frequency follows PN (or newtonian)
  // values
  const auto sep = make_with_value<DataType>(
      time, pow(initial_separation_fourth_power_, 0.25));
  return position_impl(time, sep);
}

template <typename DataType>
std::array<tnsr::I<DataType, 3>, 2> BinaryTrajectories::position_impl(
    const DataType& time, const DataType& separation) const {
  std::array<tnsr::I<DataType, 3>, 2> result{};
  const DataType orbital_freq = orbital_frequency(time);
  get<0>(result[0]) = 0.5 * separation * cos(orbital_freq * time);
  get<1>(result[0]) = 0.5 * separation * sin(orbital_freq * time);
  get<0>(result[1]) = -get<0>(result[0]);
  get<1>(result[1]) = -get<1>(result[0]);
  // Apply COM velocity
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      gsl::at(result, i).get(j) += gsl::at(center_of_mass_velocity_, j) * time;
    }
  }
  get<2>(result[0]) = center_of_mass_velocity_[2] * time;
  get<2>(result[1]) = get<2>(result[0]);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template DTYPE(data) BinaryTrajectories::separation(const DTYPE(data)&) \
      const;                                                              \
  template DTYPE(data)                                                    \
      BinaryTrajectories::orbital_frequency(const DTYPE(data)&) const;    \
  template DTYPE(data)                                                    \
      BinaryTrajectories::angular_velocity(const DTYPE(data)&) const;     \
  template std::array<tnsr::I<DTYPE(data), 3>, 2>                         \
  BinaryTrajectories::positions(const DTYPE(data)&) const;                \
  template std::array<tnsr::I<DTYPE(data), 3>, 2>                         \
  BinaryTrajectories::positions_no_expansion(const DTYPE(data)&) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
