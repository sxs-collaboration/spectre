// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Domain/Creators/TimeDependent/TestHelpers.hpp"

#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace TestHelpers::domain::creators {
void write_volume_data(
    const std::string& filename, const std::string& subfile_name,
    const std::unordered_map<
        std::string,
        std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  h5::H5File<h5::AccessType::ReadWrite> h5_file{filename, true};
  auto& vol_file = h5_file.insert<h5::VolumeData>(subfile_name);

  const auto serialized_fots =
      functions_of_time.empty() ? std::nullopt
                                : std::optional{serialize(functions_of_time)};
  // We don't care about the volume data here, just the functions of time
  vol_file.write_volume_data(
      0, 0.0,
      {ElementVolumeData{"blah",
                         {TensorComponent{"RandomTensor", DataVector{3, 0.0}}},
                         {3},
                         {Spectral::Basis::Legendre},
                         {Spectral::Quadrature::GaussLobatto}}},
      std::nullopt, serialized_fots, serialized_fots);
}
}  // namespace TestHelpers::domain::creators
