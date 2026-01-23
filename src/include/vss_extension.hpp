#pragma once

#include "duckdb/main/extension.hpp"

namespace duckdb {

class VssExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
};

} // namespace duckdb
