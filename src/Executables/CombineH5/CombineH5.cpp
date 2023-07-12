// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5PropertiesMatch.hpp"
#include "IO/H5/CombineH5.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StdHelpers.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

/*
 * This executable is used for combining a series of HDF5 volume files into one
 * continuous dataset to be stored in a single HDF5 volume file.
 */
int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;

  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "file_prefix", boost::program_options::value<std::string>()->required(),
      "prefix of the files to be combined (omit number and file extension)")(
      "subfile_name", boost::program_options::value<std::string>()->required(),
      "subfile name shared for each volume file in each H5 file (omit file "
      "extension)")("output",
                    boost::program_options::value<std::string>()->required(),
                    "combined output filename (omit file extension)");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("file_prefix") == 0u or
      vars.count("subfile_name") == 0u or vars.count("output") == 0u) {
    Parallel::printf("%s\n", desc);
    return 1;
  }

  h5::combine_h5(vars["file_prefix"].as<std::string>(),
                 vars["subfile_name"].as<std::string>(),
                 vars["output"].as<std::string>());
}
