// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_functions.h"
#include "string_tensor.h"

OrtStatusPtr string_join(const ortc::Tensor<std::string>& input_X,
                         std::string_view input_sep,
                         int64_t axis,
                         ortc::Tensor<std::string>& output) {
  OrtStatusPtr status = nullptr;
  // Setup inputs
  auto& dimensions = input_X.Shape();
  std::vector<int64_t> dimensions_out(dimensions.size() > 1 ? dimensions.size() - 1 : 1);
  int64_t size = std::accumulate(dimensions_out.begin(), dimensions_out.end(), 1ULL, std::multiplies<int64_t>());
  std::vector<std::string> out(static_cast<size_t>(size));
  output.SetStringOutput(out, dimensions_out);
  return status;
}
