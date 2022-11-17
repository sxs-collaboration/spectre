// Distributed under the MIT License.
// See LICENSE.txt for details.

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"
extern "C" void CkRegisterMainModule(void) {}
#pragma GCC diagnostic pop
