#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

struct ArrayAvgFunction {
	static void Register(ExtensionLoader &loader);
};

} // namespace duckdb
