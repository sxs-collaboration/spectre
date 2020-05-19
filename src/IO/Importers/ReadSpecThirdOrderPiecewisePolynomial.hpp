// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Importers/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace importers {
namespace Actions {
/// \brief Import SpEC `FunctionOfTime` data from an H5 file.
///
/// Uses:
///  - DataBox:
///    - `importers::Tags::FunctionOfTimeFile`
///    - `importers::Tags::FunctionOfTimeNameMap`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `::domain::Tags::FunctionsOfTime`
///
/// Columns in the file to be read must have the following form:
///   - 0 = time
///   - 1 = time of last update
///   - 2 = number of components
///   - 3 = maximum derivative order
///   - 4 = version
///   - 5 = function
///   - 6 = d/dt (function)
///   - 7 = d^2/dt^2 (function)
///   - 8 = d^3/dt^3 (function)
///
/// If the function has more than one component, columns 5-8 give
/// the first component and its derivatives, columns 9-12 give the second
/// component and its derivatives, etc.
///
struct ReadSpecThirdOrderPiecewisePolynomial {
  using const_global_cache_tags =
       tmpl::list<importers::Tags::FunctionOfTimeFile,
                  importers::Tags::FunctionOfTimeNameMap>;
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<db::tag_is_retrievable_v<importers::Tags::FunctionOfTimeFile,
                                        db::DataBox<DbTagsList>> and
               db::tag_is_retrievable_v<importers::Tags::FunctionOfTimeNameMap,
                                        db::DataBox<DbTagsList>> and
               db::tag_is_retrievable_v<::domain::Tags::FunctionsOfTime,
                                        db::DataBox<DbTagsList>>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const std::string& file_name{
        db::get<importers::Tags::FunctionOfTimeFile>(box)};

    // Get the map of SpEC -> SpECTRE FunctionofTime names
    const std::map<std::string, std::string>& dataset_name_map{
        db::get<importers::Tags::FunctionOfTimeNameMap>(box)};

    // Currently, only support order 3 piecewise polynomials.
    // This could be generalized later, but the SpEC functions of time
    // that we will read in with this action will always be 3rd-order
    // piecewise polynomials
    constexpr size_t max_deriv{3};

    std::unordered_map<std::string,
                       domain::FunctionsOfTime::PiecewisePolynomial<max_deriv>>
        spec_functions_of_time;

    std::unique_ptr<h5::H5File<h5::AccessType::ReadOnly>> file =
        std::make_unique<h5::H5File<h5::AccessType::ReadOnly>>(file_name);
    for (const auto& spec_and_spectre_names : dataset_name_map) {
      const std::string& spec_name = std::get<0>(spec_and_spectre_names);
      const std::string& spectre_name = std::get<1>(spec_and_spectre_names);
      // clang-tidy: use auto when initializing with a template cast to avoid
      // duplicating the type name
      const h5::Dat& dat_file = file->get<h5::Dat>("/" + spec_name);  // NOLINT
      const Matrix& dat_data = dat_file.get_data();

      // Check that the data in the file uses deriv order 3
      // Column 3 of the file contains the derivative order
      const size_t dat_max_deriv = dat_data(0, 3);
      if (dat_max_deriv != max_deriv) {
        file.reset();
        ERROR("Deriv order in " << file_name << " should be " << max_deriv
                                << ", not " << dat_max_deriv);
      }

      // Get the initial time ('time of last update') from the file
      // and the values of the function and its derivatives at that time
      const double start_time = dat_data(0, 1);

      // Currently, assume the same number of components are used
      // at each time. This could be generalized if needed
      const size_t number_of_components = dat_data(0, 2);

      std::array<DataVector, max_deriv + 1> initial_coefficients;
      for (size_t deriv_order = 0; deriv_order < max_deriv + 1; ++deriv_order) {
        gsl::at(initial_coefficients, deriv_order) =
            DataVector(number_of_components);
        for (size_t component = 0; component < number_of_components;
             ++component) {
          gsl::at(initial_coefficients, deriv_order)[component] =
              dat_data(0, 5 + (max_deriv + 1) * component + deriv_order);
        }
      }
      spec_functions_of_time[spectre_name] =
          domain::FunctionsOfTime::PiecewisePolynomial<3>(start_time,
                                                          initial_coefficients);

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
                dat_data(row, 4 + (max_deriv + 1) * a + max_deriv);
          }
          spec_functions_of_time[spectre_name].update(time_last_updated,
                                                      highest_derivative);
        } else {
          file.reset();
          ERROR("Non-monotonic time found in FunctionOfTime data. "
                << "Time " << dat_data(row, 1) << " follows time "
                << time_last_updated << " while reading " << spectre_name
                << "\n");
        }
      }
    }

    // Mutate ::domain::Tags::FunctionsOfTime, adding the imported
    // FunctionsOfTime to it
    db::mutate<::domain::Tags::FunctionsOfTime>(
        make_not_null(&box),
        [&dataset_name_map, &spec_functions_of_time,
         &file](const gsl::not_null<std::unordered_map<
                    std::string,
                    std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
                    functions_of_time) {
          for (auto spec_and_spectre_names : dataset_name_map) {
            const std::string& spectre_name =
                std::get<1>(spec_and_spectre_names);

            // The FunctionsOfTime we are mutating must already have
            // an element with key==spectre_name; this action only
            // mutates the value associated with that key
            if ((*functions_of_time).count(spectre_name) == 0) {
              file.reset();
              ERROR("Trying to import data for key "
                    << spectre_name
                    << "in FunctionsOfTime, but FunctionsOfTime does not "
                       "contain that key.\n");
            }
            auto* piecewise_polynomial = dynamic_cast<
                domain::FunctionsOfTime::PiecewisePolynomial<max_deriv>*>(
                (*functions_of_time)[spectre_name].get());
            if (piecewise_polynomial == nullptr) {
              file.reset();
              ERROR(
                  "The function of time with name spectre_name is not a "
                  "PiecewisePolynomial<"
                  << max_deriv
                  << "> and so cannot be set using "
                     "ReadSpecThirdOrderPiecewisePolynomial\n");
            }
            *piecewise_polynomial = spec_functions_of_time[spectre_name];
          }
        });
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace importers
