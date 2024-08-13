// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <csignal>

#ifndef SPECTRE_BLAZE_EXCEPTIONS_HPP
#define SPECTRE_BLAZE_EXCEPTIONS_HPP

#ifdef SPECTRE_DEBUG
#define BLAZE_THROW(EXCEPTION)           \
  struct sigaction handler {};           \
  handler.sa_handler = SIG_IGN;          \
  handler.sa_flags = 0;                  \
  sigemptyset(&handler.sa_mask);         \
  sigaction(SIGTRAP, &handler, nullptr); \
  raise(SIGTRAP);                        \
  throw EXCEPTION
#else  // SPECTRE_DEBUG
#define BLAZE_THROW(EXCEPTION)
#endif  // SPECTRE_DEBUG
#endif  // BLAZE_EXCEPTIONSSPECTRE_BLAZE_EXCEPTIONS_HPP
