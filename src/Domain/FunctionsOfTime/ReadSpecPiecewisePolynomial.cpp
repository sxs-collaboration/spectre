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
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::FunctionsOfTime {
template <template <size_t> class FoTType, size_t MaxDeriv>
void read_spec_piecewise_polynomial(
    const gsl::not_null<std::unordered_map<std::string, FoTType<MaxDeriv>>*>
        spec_functions_of_time,
    const std::string& file_name,
    const std::map<std::string, std::string>& dataset_name_map,
    const bool quaternion_rotation) {
  h5::H5File<h5::AccessType::ReadOnly> file{file_name};
  for (const auto& name_pair : dataset_name_map) {
    const auto& spec_name = name_pair.first;
    const auto& spectre_name = name_pair.second;
    file.close_current_object();
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

    if (quaternion_rotation) {
      ASSERT(number_of_components == 1 or number_of_components == 4,
             "To read in a function of time representing rotation from SpEC, "
             "it must have either 1 component representing rotation about the "
             "z-axis or 4 components representing the 'quaternion' (which is "
             "actually only the quaternion for the 0th deriv order).");
    }

    std::array<DataVector, MaxDeriv + 1> initial_coefficients;
    for (size_t deriv_order = 0; deriv_order < MaxDeriv + 1; ++deriv_order) {
      if (not quaternion_rotation) {
        gsl::at(initial_coefficients, deriv_order) =
            DataVector(number_of_components);
      } else {
        // Quaternion rotation will always need 3 components. The
        // `number_of_components` variable represents the number of components
        // in the SpEC file and can be either 1 or 4 as explained above. When
        // creating the `initial_coefficients` for a QuaternionFunctionOfTime,
        // they will be used as initial values for the internal angle
        // PiecewisePolynomial inside the QuaternionFunctionOfTime. This is why
        // we need 3 componenets here, for rotations about x,y,z.
        gsl::at(initial_coefficients, deriv_order) = DataVector(3, 0.0);
      }
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
        if (not quaternion_rotation) {
          gsl::at(initial_coefficients, deriv_order)[component] =
              dat_data(0, 5 + (MaxDeriv + 1) * component + deriv_order);
        } else {
          if (number_of_components == 1) {
            // z-component
            gsl::at(initial_coefficients, deriv_order)[2] =
                dat_data(0, 5 + (MaxDeriv + 1) * component + deriv_order);
          } else {
            // quaternion. Since the 0th derivative represents the quaternion
            // and not a rotation angle, set the 0th deriv components to 0.0 to
            // start. Then for the nth derivative, the 0th component is
            // meaningless and components 1-3 are what we are after
            if (deriv_order == 0 or component == 0) {
              // 0th deriv order is already initialized to 0
              // 0th component is meaningless in SpEC so don't use it
              continue;
            } else {
              // for nth deriv order and not 0th component fill in values as is
              gsl::at(initial_coefficients, deriv_order)[component - 1] =
                  dat_data(0, 5 + (MaxDeriv + 1) * component + deriv_order);
            }
          }
        }
      }
    }
    if constexpr (std::is_same_v<FoTType<MaxDeriv>,
                                 domain::FunctionsOfTime::
                                     QuaternionFunctionOfTime<MaxDeriv>>) {
      if (quaternion_rotation) {
        (*spec_functions_of_time)[spectre_name] =
            domain::FunctionsOfTime::QuaternionFunctionOfTime<MaxDeriv>(
                start_time, std::array<DataVector, 1>{{{1.0, 0.0, 0.0, 0.0}}},
                initial_coefficients, start_time);
      } else {
        // Shouldn't get here, but just in case
        ERROR(
            "Trying to use a QuaternionFunctionOfTime when not using "
            "quaternion rotation. Trying to set QuaternionFunctionOfTime for "
            "spectre name '"
            << spectre_name << "'.");
      }
    } else {
      (*spec_functions_of_time)[spectre_name] =
          domain::FunctionsOfTime::PiecewisePolynomial<MaxDeriv>(
              start_time, initial_coefficients, start_time);
    }

    // Loop over the remaining times, updating the function of time
    DataVector highest_derivative{};
    if (not quaternion_rotation) {
      highest_derivative = DataVector{number_of_components, 0.0};
    } else {
      // Quaternion rotation always has 3 components
      highest_derivative =
          DataVector{3, std::numeric_limits<double>::signaling_NaN()};
    }
    double time_last_updated = start_time;
    for (size_t row = 1; row < dat_data.rows(); ++row) {
      // If time of last update has changed, then update the FunctionOfTime
      // The time of last update is stored in column 1 in the dat file
      if (dat_data(row, 1) > time_last_updated) {
        time_last_updated = dat_data(row, 1);
        for (size_t a = 0; a < number_of_components; ++a) {
          if (not quaternion_rotation) {
            highest_derivative[a] =
                dat_data(row, 5 + (MaxDeriv + 1) * a + MaxDeriv);
          } else {
            if (number_of_components == 1) {
              // z-rotation
              highest_derivative[0] = 0.0;
              highest_derivative[1] = 0.0;
              highest_derivative[2] =
                  dat_data(row, 5 + (MaxDeriv + 1) * a + MaxDeriv);
            } else {
              // quaternion
              if (a == 0) {
                // 0th component is meaningless in SpEC so don't use it
                continue;
              } else {
                // fill in a != 0 as is
                highest_derivative[a - 1] =
                    dat_data(row, 5 + (MaxDeriv + 1) * a + MaxDeriv);
              }
            }
          }
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

void override_functions_of_time(
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        functions_of_time,
    const std::string& function_of_time_file,
    const std::map<std::string, std::string>& function_of_time_name_map) {
  std::unordered_map<std::string,
                     domain::FunctionsOfTime::PiecewisePolynomial<2>>
      spec_functions_of_time_second_order{};
  std::unordered_map<std::string,
                     domain::FunctionsOfTime::PiecewisePolynomial<3>>
      spec_functions_of_time_third_order{};
  std::unordered_map<std::string,
                     domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>
      spec_functions_of_time_quaternion{};

  // Import those functions of time of each supported order
  domain::FunctionsOfTime::read_spec_piecewise_polynomial(
      make_not_null(&spec_functions_of_time_second_order),
      function_of_time_file, function_of_time_name_map);
  domain::FunctionsOfTime::read_spec_piecewise_polynomial(
      make_not_null(&spec_functions_of_time_third_order), function_of_time_file,
      function_of_time_name_map);

  bool uses_quaternion_rotation = false;
  for (const auto& name_and_fot : *functions_of_time) {
    auto* maybe_quaternion_fot =
        dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
            name_and_fot.second.get());
    if (maybe_quaternion_fot != nullptr) {
      uses_quaternion_rotation = true;
    }
  }

  // Only parse as quaternion function of time if it exists
  if (uses_quaternion_rotation) {
    domain::FunctionsOfTime::read_spec_piecewise_polynomial(
        make_not_null(&spec_functions_of_time_quaternion),
        function_of_time_file, function_of_time_name_map, true);
  }

  for (const auto& name_pair : function_of_time_name_map) {
    const auto& spec_name = name_pair.first;
    const auto& spectre_name = name_pair.second;
    (void)spec_name;
    // The FunctionsOfTime we are mutating must already have
    // an element with key==spectre_name; this action only
    // mutates the value associated with that key
    if (functions_of_time->count(spectre_name) == 0) {
      ERROR("Trying to import data for key "
            << spectre_name
            << " in FunctionsOfTime, but FunctionsOfTime does not "
               "contain that key. This might happen if the option "
               "FunctionOfTimeNameMap is not specified correctly. Keys "
               "contained in FunctionsOfTime: "
            << keys_of(*functions_of_time) << "\n");
    }
    auto* piecewise_polynomial_second_order =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            (*functions_of_time)[spectre_name].get());
    auto* piecewise_polynomial_third_order =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<3>*>(
            (*functions_of_time)[spectre_name].get());
    auto* quaternion_fot_third_order =
        dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
            (*functions_of_time)[spectre_name].get());
    if (piecewise_polynomial_second_order == nullptr) {
      if (piecewise_polynomial_third_order == nullptr) {
        if (quaternion_fot_third_order == nullptr) {
          ERROR("The function of time with name "
                << spectre_name
                << " is not a PiecewisePolynomial<2>, "
                   "PiecewisePolynomial<3>, or QuaternionFunctionOfTime<3> "
                   "and so cannot be set using "
                   "read_spec_piecewise_polynomial\n");
        } else {
          *quaternion_fot_third_order =
              spec_functions_of_time_quaternion.at(spectre_name);
        }
      } else {
        *piecewise_polynomial_third_order =
            spec_functions_of_time_third_order.at(spectre_name);
      }
    } else {
      *piecewise_polynomial_second_order =
          spec_functions_of_time_second_order.at(spectre_name);
    }
  }
}
}  // namespace domain::FunctionsOfTime

#define FOTTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template void domain::FunctionsOfTime::read_spec_piecewise_polynomial<>( \
      const gsl::not_null<std::unordered_map<std::string, FOTTYPE(data)>*> \
          spec_functions_of_time,                                          \
      const std::string& file_name,                                        \
      const std::map<std::string, std::string>& dataset_name_map,          \
      const bool quaternion_rotation);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (domain::FunctionsOfTime::PiecewisePolynomial<2>,
                         domain::FunctionsOfTime::PiecewisePolynomial<3>,
                         domain::FunctionsOfTime::QuaternionFunctionOfTime<3>))

#undef FOTTYPE
#undef INSTANTIATE
