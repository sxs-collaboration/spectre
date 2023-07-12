// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <boost/program_options.hpp>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5PropertiesMatch.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
void combine_h5(const std::string& file_prefix, const std::string& subfile_name,
                const std::string& output);
}  // namespace h5
