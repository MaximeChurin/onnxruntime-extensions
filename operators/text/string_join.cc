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
  auto& X = input_X.Data();
  auto& dimensions = input_X.Shape();

  std::vector<int64_t> dimensions_out(dimensions.size() > 1 ? dimensions.size() - 1 : 1);
  for (size_t i = 0, pos = 0; i < dimensions.size(); ++i) {
    if (static_cast<int64_t>(i) == axis)
      continue;
    dimensions_out[pos++] = dimensions[i];
  }

  int64_t size = std::accumulate(dimensions_out.begin(), dimensions_out.end(), 1ULL, std::multiplies<int64_t>());
  std::vector<std::string> out(static_cast<size_t>(size));

  // Do computation
  int64_t h = 1;
  for (size_t i = static_cast<size_t>(axis + 1); i < dimensions.size(); ++i) {
    h *= dimensions[i];
  }
  int64_t left_part = size / h;
  int64_t right_part = size / left_part;
  int64_t n_red = dimensions[static_cast<size_t>(axis)] - 1;
  int64_t inc = right_part * (n_red + 1);
  int64_t pos = 0;
  for (int64_t li = 0; li < left_part; ++li) {
    for (int64_t ri = 0; ri < right_part; ++ri, ++pos) {
      std::ostringstream st;
      int64_t index = ri + li * inc;
      for (int64_t j = 0; j < n_red; ++j, index += h) {
        st << X[static_cast<size_t>(index)] << input_sep;
      }
      st << X[static_cast<size_t>(index)];
      out[static_cast<size_t>(pos)] = st.str();
    }
  }
  output.SetStringOutput(out, dimensions_out);
  return status;
}
