// Distributed under the MIT License.
// Copyright (c) 2009-2017 by the contributors to libc++, listed at
// https://llvm.org/svn/llvm-project/libcxx/trunk/CREDITS.TXT

// This is a portion of
// https://llvm.org/svn/llvm-project/libcxx/trunk/test/std/containers/sequences/array/begin.pass.cpp

#include <array>
#include <cassert>

int main() {
  struct NoDefault {
    NoDefault(int) {}
  };
  typedef NoDefault T;
  typedef std::array<T, 0> C;
  C c = {};
  assert(c.begin() == c.end());
}
