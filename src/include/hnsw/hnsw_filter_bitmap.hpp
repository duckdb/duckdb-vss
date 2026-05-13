#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/storage_index.hpp"

namespace duckdb {

class ClientContext;
class TableCatalogEntry;

//! A dense boolean bitmap indexed by row_t. Used to pre-filter HNSW searches.
//! Rows not present in the bitmap are excluded from graph traversal so they do
//! not consume the `wanted` budget, avoiding post-filter underfill.
//!
//! Uses vector<uint8_t> (one byte per slot) rather than vector<bool> (bit-packed)
//! to avoid bit-manipulation overhead on the hot predicate path inside USearch's
//! linear filtered scan.
struct FilterBitmap {
	vector<uint8_t> bits; // 1 byte per row_id slot: 0 = absent, 1 = present

	bool Contains(row_t row_id) const noexcept {
		if (row_id < 0) {
			return false;
		}
		auto idx = static_cast<idx_t>(row_id);
		return idx < bits.size() && bits[idx] != 0;
	}

	void Set(row_t row_id) {
		if (row_id < 0) {
			return;
		}
		auto idx = static_cast<idx_t>(row_id);
		if (idx >= bits.size()) {
			bits.resize(idx + 1, 0);
		}
		bits[idx] = 1;
	}
};

//! Scan the filter columns of `table` for all transaction-visible rows that
//! satisfy `filters` and mark their row_ids in the returned bitmap.
//!
//! @param logical_col_ids  Logical column indices that are keys in filters.filters
//! @param col_types        Types of those columns (positionally aligned)
unique_ptr<FilterBitmap> BuildFilterBitmap(ClientContext &context, TableCatalogEntry &table,
                                           const TableFilterSet &filters,
                                           const vector<idx_t> &logical_col_ids,
                                           const vector<LogicalType> &col_types);

} // namespace duckdb
