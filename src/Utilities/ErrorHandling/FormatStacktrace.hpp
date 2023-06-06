// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Formatter for Boost stacktrace, following these docs:
/// https://www.boost.org/doc/libs/1_78_0/doc/html/stacktrace/getting_started.html#stacktrace.getting_started.global_control_over_stacktrace_o

#pragma once

#include <boost/stacktrace/stacktrace_fwd.hpp>
#include <ostream>

std::ostream& operator<<(std::ostream& os,
                         const boost::stacktrace::stacktrace& backtrace);
