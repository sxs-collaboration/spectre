// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ReductionActions.hpp"

#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace observers {
void DatRows::pup(PUP::er& p) {
  p | subfile_name;
  p | legend;
  p | rows;
}

bool operator==(const DatRows& lhs, const DatRows& rhs) {
  return lhs.subfile_name == rhs.subfile_name and lhs.legend == rhs.legend and
         lhs.rows == rhs.rows;
}

bool operator!=(const DatRows& lhs, const DatRows& rhs) {
  return not(lhs == rhs);
}
}  // namespace observers

namespace observers::ThreadedActions::ReductionActions_detail {

void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const double t) {
  all_reduction_data->push_back(t);
}

template <typename T>
void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const std::vector<T>& t) {
  all_reduction_data->insert(all_reduction_data->end(), t.begin(), t.end());
}

void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const std::array<double, 3>& t) {
  all_reduction_data->insert(all_reduction_data->end(), t.begin(), t.end());
}

// Generate instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                  \
  template void append_to_reduction_data<DTYPE(data)>(        \
      gsl::not_null<std::vector<double>*> all_reduction_data, \
      const std::vector<DTYPE(data)>& t);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, size_t))

#undef DTYPE
#undef INSTANTIATE

void write_rows(const std::vector<DatRows>& data,
                const std::string& input_source,
                const std::string& file_prefix) {
  // Check all rows before opening the file so an error leaves the file
  // untouched.
  for (const auto& [subfile_name, legend, rows] : data) {
    for (const auto& row : rows) {
      if (row.size() != legend.size()) {
        ERROR(
            "There must be one name provided for each piece of data. You "
            "provided "
            << legend.size() << " names: '" << get_output(legend)
            << "' but a row for subfile '" << subfile_name << "' has "
            << row.size() << " pieces of data");
      }
    }
  }

  h5::H5File<h5::AccessType::ReadWrite> h5file(file_prefix + ".h5", true,
                                               input_source);
  constexpr size_t version_number = 0;
  for (const auto& [subfile_name, legend, rows] : data) {
    auto& time_series_file =
        h5file.try_insert<h5::Dat>(subfile_name, legend, version_number);
    time_series_file.append(rows);
    h5file.close_current_object();
  }
}
}  // namespace observers::ThreadedActions::ReductionActions_detail
