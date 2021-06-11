// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/ReadSpecPiecewisePolynomial.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::FunctionsOfTime {
template <size_t MaxDeriv>
void read_spec_piecewise_polynomial(
    const gsl::not_null<std::unordered_map<
        std::string, domain::FunctionsOfTime::PiecewisePolynomial<MaxDeriv>>*>
        spec_functions_of_time,
    const std::string& file_name,
    const std::map<std::string, std::string>& dataset_name_map) noexcept {
  h5::H5File<h5::AccessType::ReadOnly> file{file_name};
  for (const auto& [spec_name, spectre_name] : dataset_name_map) {
    const auto& dat_file = file.get<h5::Dat>("/" + spec_name);
    const auto& dat_data = dat_file.get_data();

    // Check that the data in the file uses deriv order MaxDeriv
    // Column 3 of the file contains the derivative order
    // If not, just continue: this function will only read in
    // FunctionsOfTime whose DerivOrder matches.
    // Note: this is not an ERROR(), because a future call to
    // this function but with different MaxDeriv might succeed.
    const size_t dat_max_deriv = dat_data(0, 3);
    if (dat_max_deriv != MaxDeriv) {
      continue;
    }

    // Get the initial time ('time of last update') from the file
    // and the values of the function and its derivatives at that time
    const double start_time = dat_data(0, 1);

    // Currently, assume the same number of components are used
    // at each time. This could be generalized if needed
    const size_t number_of_components = dat_data(0, 2);

    std::array<DataVector, MaxDeriv + 1> initial_coefficients;
    for (size_t deriv_order = 0; deriv_order < MaxDeriv + 1; ++deriv_order) {
      gsl::at(initial_coefficients, deriv_order) =
          DataVector(number_of_components);
      for (size_t component = 0; component < number_of_components;
           ++component) {
        // Columns in the file to be read have the following form. Columns
        // 0 through 4 contain the following quantities:
        // 0 == time, 1 == time of last update, 2 == number of components,
        // 3 == maximum derivative order, 4 == version.
        // After this, each component takes up MaxDeriv + 1 columns, which are
        // the zeroth, first, second, ... MaxDeriv-th time derivatives of the
        // component. The nth-order derivative of the ith component is therefore
        // in column 5 + (MaxDeriv + 1) * i + n.
        gsl::at(initial_coefficients, deriv_order)[component] =
            dat_data(0, 5 + (MaxDeriv + 1) * component + deriv_order);
      }
    }
    (*spec_functions_of_time)[spectre_name] =
        domain::FunctionsOfTime::PiecewisePolynomial<MaxDeriv>(
            start_time, initial_coefficients, start_time);

    // Loop over the remaining times, updating the function of time
    DataVector highest_derivative(number_of_components);
    double time_last_updated = start_time;
    for (size_t row = 1; row < dat_data.rows(); ++row) {
      // If time of last update has changed, then update the FunctionOfTime
      // The time of last update is stored in column 1 in the dat file
      if (dat_data(row, 1) > time_last_updated) {
        time_last_updated = dat_data(row, 1);
        for (size_t a = 0; a < number_of_components; ++a) {
          highest_derivative[a] =
              dat_data(row, 5 + (MaxDeriv + 1) * a + MaxDeriv);
        }
        (*spec_functions_of_time)[spectre_name].update(
            time_last_updated, highest_derivative, time_last_updated);
      } else {
        ERROR("Non-monotonic time found in FunctionOfTime data. "
              << "Time " << dat_data(row, 1) << " follows time "
              << time_last_updated << " while reading " << spectre_name
              << "\n");
      }
      // Column 2 (number of components), column 3 (max deriv order), and
      // column 4 (version) should not change
      if (dat_data(row, 2) != dat_data(0, 2) or
          dat_data(row, 3) != dat_data(0, 3) or
          dat_data(row, 4) != dat_data(0, 4)) {
        ERROR(
            "Values in column 2 (number of components), column 3 (maximum "
            "derivative order), or column 4 (version) in SpEC history files "
            "should be the same in each row. But columns (2, 3, 4) in row "
            << row << " have values (" << dat_data(row, 2) << ", "
            << dat_data(row, 3) << ", " << dat_data(row, 4)
            << "), while in row 0 they have values (" << dat_data(0, 2) << ", "
            << dat_data(0, 3) << ", " << dat_data(0, 4) << ")");
      }
    }
  }
}
}  // namespace domain::FunctionsOfTime

#define MAX_DERIV(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                \
  template void                                                             \
  domain::FunctionsOfTime::read_spec_piecewise_polynomial<MAX_DERIV(data)>( \
      const gsl::not_null<std::unordered_map<                               \
          std::string,                                                      \
          domain::FunctionsOfTime::PiecewisePolynomial<MAX_DERIV(data)>>*>  \
          spec_functions_of_time,                                           \
      const std::string& file_name,                                         \
      const std::map<std::string, std::string>& dataset_name_map);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef MAX_DERIV
#undef INSTANTIATE
