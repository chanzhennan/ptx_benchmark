#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "op.h"

PYBIND11_MODULE(gpumark, m) {
  //////////////////////////////   ops   //////////////////////////////
  // ops: decode attention
  m.def("AddPi", &AddPi);
}
