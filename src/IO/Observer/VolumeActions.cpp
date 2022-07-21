// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/VolumeActions.hpp"

#include <string>
#include <vector>

#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace observers::ThreadedActions::VolumeActions_detail {
void write_data(const std::string& h5_file_name,
                const std::string& input_source,
                const std::string& subfile_path,
                const observers::ObservationId& observation_id,
                std::vector<ElementVolumeData>&& volume_data) {
  const uint32_t version_number = 0;
  {
    h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name + ".h5"s, true,
                                                  input_source};
    auto& volume_file =
        h5_file.try_insert<h5::VolumeData>(subfile_path, version_number);
    volume_file.write_volume_data(observation_id.hash(), observation_id.value(),
                                  volume_data);
  }
}
}  // namespace observers::ThreadedActions::VolumeActions_detail
