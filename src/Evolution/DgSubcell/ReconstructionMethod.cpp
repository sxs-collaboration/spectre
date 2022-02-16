// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/ReconstructionMethod.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

namespace evolution::dg::subcell::fd {
std::ostream& operator<<(std::ostream& os, ReconstructionMethod recons_method) {
  switch (recons_method) {
    case ReconstructionMethod::DimByDim:
      return os << "DimByDim";
    case ReconstructionMethod::AllDimsAtOnce:
      return os << "AllDimsAtOnce";
    default:
      ERROR("Unknown reconstruction method, must be DimByDim or AllDimsAtOnce");
  }
}
}  // namespace evolution::dg::subcell::fd

template <>
evolution::dg::subcell::fd::ReconstructionMethod
Options::create_from_yaml<evolution::dg::subcell::fd::ReconstructionMethod>::
    create<void>(const Options::Option& options) {
  const auto recons_method = options.parse_as<std::string>();
  if (recons_method == get_output(type::DimByDim)) {
    return type::DimByDim;
  } else if (recons_method == get_output(type::AllDimsAtOnce)) {
    return type::AllDimsAtOnce;
  } else {
    PARSE_ERROR(options.context(),
                "ReconstructionMethod must be '"
                    << get_output(type::DimByDim) << "', or '"
                    << get_output(type::AllDimsAtOnce) << "'");
  }
}
