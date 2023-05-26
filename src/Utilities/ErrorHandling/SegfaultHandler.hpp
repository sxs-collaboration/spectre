// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functions to enable handling segmentation faults

#pragma once

/// \ingroup ErrorHandlingGroup
/// After a call to this function, the code will handle `SIGSEGV` segmentation
/// faults by printing an error with a stacktrace.
void enable_segfault_handler();
