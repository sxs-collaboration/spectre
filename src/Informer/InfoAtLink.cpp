// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is compiled into every executable at link time, so the information
// that these functions provide reflects the time the executable was linked
// instead of the time that CMake was last run.

#include "Informer/InfoFromBuild.hpp"

#include <boost/preprocessor.hpp>
#include <string>

std::string link_date() { return std::string(__TIMESTAMP__); }

std::string git_description() {
  return std::string(BOOST_PP_STRINGIZE(GIT_DESCRIPTION));
}

std::string git_branch() { return std::string(BOOST_PP_STRINGIZE(GIT_BRANCH)); }
