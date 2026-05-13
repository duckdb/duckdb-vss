#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

// Forward declarations — defined after HNSWIndexScanOptimizer
static void FoldLateMaterializationJoins(unique_ptr<LogicalOperator> &plan);
static void EliminateRedundantOrders(unique_ptr<LogicalOperator> &plan);

//-----------------------------------------------------------------------------
// Plan rewriter
//-----------------------------------------------------------------------------
class HNSWIndexScanOptimizer : public OptimizerExtension {
public:
	HNSWIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		if (top_n.orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = top_n.orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		// find the expression that is referenced
		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();

		// This the expression that is referenced by the order by expression
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		// The projection must sit on top of a get
		if (projection.children.size() != 1 || projection.children.front()->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = projection.children.front();
		auto &get = get_ptr->Cast<LogicalGet>();
		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			// We can only replace the scan if the table is a duck table
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		// Find the index
		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;

		table_info.BindIndexes(context, HNSWIndex::TYPE_NAME);
		table_info.GetIndexes().Scan([&](Index &index) {
			if (!index.IsBound() || HNSWIndex::TYPE_NAME != index.GetIndexType()) {
				return false;
			}
			auto &cast_index = index.Cast<HNSWIndex>();

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!cast_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				return false;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				return false;
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					return false;
				}
			}

			const auto vector_size = cast_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}

			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, cast_index, top_n.limit, std::move(query_vector));
			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}

		// If there are no table filters pushed down into the get, we can just replace the get with the index scan
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = HNSWIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);
		if (get.table_filters.filters.empty()) {
			// Remove the TopN operator
			plan = std::move(top_n.children[0]);
			return true;
		}

		// Check whether filter pushdown into the HNSW scan is enabled
		Value pushdown_enabled_val;
		bool use_pushdown = true;
		if (context.TryGetCurrentSetting("hnsw_enable_filter_pushdown", pushdown_enabled_val)) {
			if (!pushdown_enabled_val.IsNull()) {
				use_pushdown = pushdown_enabled_val.GetValue<bool>();
			}
		}

		auto &hnsw_bind = get.bind_data->Cast<HNSWIndexScanBindData>();

		if (use_pushdown) {
			// Push filters into the bind data so the scan builds a FilterBitmap at execution time.
			// This ensures filtered candidates don't consume the wanted budget in filtered_ef_search.
			for (const auto &entry : get.table_filters.filters) {
				hnsw_bind.filter_logical_col_ids.push_back(entry.first);
				hnsw_bind.filter_col_types.push_back(get.returned_types[entry.first]);
			}
			hnsw_bind.pushed_filters = get.table_filters.Copy();
			// Clear from LogicalGet so DuckDB does not create a duplicate LogicalFilter above.
			get.table_filters.filters.clear();
		} else {
			// Escape hatch: fall back to the original post-filter pull-up behavior.
			// Keep the TopN in the plan so LIMIT/ORDER BY are preserved.
			get.projection_ids.clear();
			get.types.clear();

			auto new_filter = make_uniq<LogicalFilter>();
			auto &column_ids = get.GetColumnIds();
			for (const auto &entry : get.table_filters.filters) {
				idx_t column_id = entry.first;
				auto &type = get.returned_types[column_id];
				bool found = false;
				for (idx_t i = 0; i < column_ids.size(); i++) {
					if (column_ids[i].GetPrimaryIndex() == column_id) {
						column_id = i;
						found = true;
						break;
					}
				}
				if (!found) {
					throw InternalException("Could not find column id for filter");
				}
				auto column =
				    make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
				new_filter->expressions.push_back(entry.second->ToExpression(*column));
			}
			new_filter->children.push_back(std::move(get_ptr));
			new_filter->ResolveOperatorTypes();
			get_ptr = std::move(new_filter);
			// Don't remove TopN here — the Filter node sits above the scan, TopN stays above everything.
			return true;
		}

		// Pushdown path: remove the TopN operator (HNSW scan already enforces the limit).
		plan = std::move(top_n.children[0]);
		return true;
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {

		auto ok = TryOptimize(context, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "hnsw_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];

						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							// The binding is not referenced
							// We can remove this expression. But positionality matters so just replace with int.
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_hnsw_scan = OptimizeChildren(input.context, plan);
		if (did_use_hnsw_scan) {
			MergeProjections(plan);
			FoldLateMaterializationJoins(plan);
			EliminateRedundantOrders(plan);
		}
	}
};

//-----------------------------------------------------------------------------
// LateMaterialization join folder
//-----------------------------------------------------------------------------
// DuckDB's LateMaterialization optimizer (which runs before extension optimizers)
// may split a query like:
//
//   SELECT id FROM t ORDER BY array_distance(vec, q) LIMIT k
//
// into:
//
//   PROJECTION(id, dist)
//     SEMI_JOIN(left.rowid = right.rowid)
//       left:  SEQ_SCAN(t, [id, vec])          ← O(n) full scan
//       right: PROJECTION → HNSW_INDEX_SCAN(t, [vec, rowid])
//
// After OUR scan optimizer runs (on the right branch), the right side already
// has an HNSW index scan.  Here we fold the SEMI JOIN away: we add the
// missing columns (e.g., `id`) directly to the HNSW scan so the O(n) left
// seq_scan can be eliminated entirely.
//-----------------------------------------------------------------------------

// Return the unique_ptr owner of the first LogicalGet(hnsw_index_scan) found in
// the subtree rooted at `root`, or nullptr if none exists.
// Stops (returns nullptr) if a LOGICAL_FILTER node is encountered — this means
// the escape-hatch path was taken and the filter must not be discarded.
static unique_ptr<LogicalOperator> *FindHNSWGetOwner(unique_ptr<LogicalOperator> &root) {
	// A filter node signals the escape hatch was used; bail out to preserve it.
	if (root->type == LogicalOperatorType::LOGICAL_FILTER) {
		return nullptr;
	}
	if (root->type == LogicalOperatorType::LOGICAL_GET &&
	    root->Cast<LogicalGet>().function.name == "hnsw_index_scan") {
		return &root;
	}
	for (auto &child : root->children) {
		auto *found = FindHNSWGetOwner(child);
		if (found) {
			return found;
		}
	}
	return nullptr;
}

// Attempt to fold a LateMaterialization SEMI JOIN into the HNSW scan.
//
// On success:
//   - Missing columns are added to hnsw_get.column_ids / hnsw_get.types.
//   - `semi_join_ref` is replaced with the HNSW GET node directly.
//   - `replacer` is populated with (left_tbl_binding → hnsw_tbl_binding) mappings
//     so the caller can fix column references in the parent operator.
static bool TryFoldSemiJoin(unique_ptr<LogicalOperator> &semi_join_ref, ColumnBindingReplacer &replacer) {
	auto &join = semi_join_ref->Cast<LogicalComparisonJoin>();
	if (join.join_type != JoinType::SEMI || join.children.size() != 2) {
		return false;
	}

	// Left child must be a seq_scan on a DuckDB table
	auto &left = join.children[0];
	if (left->type != LogicalOperatorType::LOGICAL_GET) {
		return false;
	}
	auto &left_get = left->Cast<LogicalGet>();
	if (left_get.function.name != "seq_scan") {
		return false;
	}
	if (!left_get.GetTable() || !left_get.GetTable()->IsDuckTable()) {
		return false;
	}

	// Validate this is DuckDB's LateMaterialization pattern: a rowid-equality semi join.
	// LateMaterialization always produces exactly one condition: left.rowid = right.rowid.
	// If the condition is anything else this is a user-written semi join we must not fold.
	if (join.conditions.size() != 1) {
		return false;
	}
	const auto &cond = join.conditions[0];
	if (cond.comparison != ExpressionType::COMPARE_EQUAL ||
	    cond.left->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	const auto &left_ref = cond.left->Cast<BoundColumnRefExpression>();
	if (left_ref.binding.table_index != left_get.table_index) {
		return false;
	}
	{
		const auto &left_col_ids = left_get.GetColumnIds();
		const idx_t output_k = left_ref.binding.column_index;
		const idx_t scan_pos =
		    left_get.projection_ids.empty() ? output_k : left_get.projection_ids[output_k];
		if (scan_pos >= left_col_ids.size() || !left_col_ids[scan_pos].IsRowIdColumn()) {
			return false;
		}
	}

	// Right subtree must contain an hnsw_index_scan for the same physical table
	auto *hnsw_owner = FindHNSWGetOwner(join.children[1]);
	if (!hnsw_owner) {
		return false;
	}
	auto &hnsw_get = (*hnsw_owner)->Cast<LogicalGet>();
	if (!hnsw_get.GetTable() ||
	    &left_get.GetTable()->GetStorage() != &hnsw_get.GetTable()->GetStorage()) {
		return false;
	}

	// Build a map: logical column ID → current scan position in hnsw_get
	unordered_map<idx_t, idx_t> logical_to_hnsw_pos;
	idx_t hnsw_rowid_pos = DConstants::INVALID_INDEX;
	{
		const auto &h_col_ids = hnsw_get.GetColumnIds();
		for (idx_t j = 0; j < h_col_ids.size(); j++) {
			if (h_col_ids[j].IsRowIdColumn()) {
				hnsw_rowid_pos = j;
			} else if (!h_col_ids[j].IsVirtualColumn()) {
				logical_to_hnsw_pos[h_col_ids[j].GetPrimaryIndex()] = j;
			}
		}
	}

	// Clear projection_ids so we manage column positions directly.
	// Types will be rebuilt from scratch once all columns are finalised.
	hnsw_get.projection_ids.clear();

	// For each output column of left_get, ensure hnsw_get projects it and
	// record the binding replacement (left_binding → hnsw_binding).
	const auto left_bindings = left_get.GetColumnBindings();
	const auto &left_col_ids = left_get.GetColumnIds();

	for (idx_t k = 0; k < left_bindings.size(); k++) {
		const auto old_binding = left_bindings[k]; // (left_tbl_idx, col_idx)

		// Resolve the physical ColumnIndex for this output slot,
		// taking into account possible projection_ids on the left scan.
		const idx_t scan_pos = left_get.projection_ids.empty() ? k : left_get.projection_ids[k];
		const auto &left_col_idx = left_col_ids[scan_pos];

		idx_t hnsw_pos;
		if (left_col_idx.IsRowIdColumn()) {
			// rowid
			if (hnsw_rowid_pos == DConstants::INVALID_INDEX) {
				hnsw_rowid_pos = hnsw_get.GetMutableColumnIds().size();
				hnsw_get.GetMutableColumnIds().emplace_back(COLUMN_IDENTIFIER_ROW_ID);
			}
			hnsw_pos = hnsw_rowid_pos;
		} else {
			auto primary = left_col_idx.GetPrimaryIndex();
			auto it = logical_to_hnsw_pos.find(primary);
			if (it != logical_to_hnsw_pos.end()) {
				hnsw_pos = it->second;
			} else {
				// Column is not yet in hnsw_get – add it
				hnsw_pos = hnsw_get.GetMutableColumnIds().size();
				hnsw_get.GetMutableColumnIds().push_back(left_col_idx);
				logical_to_hnsw_pos[primary] = hnsw_pos;
			}
		}

		replacer.replacement_bindings.emplace_back(old_binding,
		                                            ColumnBinding(hnsw_get.table_index, hnsw_pos));
	}

	// Rebuild output types to match the (now-final) column_ids
	hnsw_get.types.clear();
	for (const auto &col_id : hnsw_get.GetColumnIds()) {
		hnsw_get.types.push_back(hnsw_get.GetColumnType(col_id));
	}

	// Extract the HNSW GET from the right branch and replace the SEMI JOIN
	auto hnsw_node = std::move(*hnsw_owner);
	semi_join_ref = std::move(hnsw_node);
	// semi_join (and the left seq_scan) are now destroyed.

	return true;
}

// Walk the plan tree.  For each direct child that is a SEMI JOIN, attempt to
// fold it.  On success, apply the column-binding replacer to the current node
// (which fixes all references to the eliminated left scan's bindings).
static void FoldLateMaterializationJoins(unique_ptr<LogicalOperator> &plan) {
	for (auto &child : plan->children) {
		if (child->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
		    child->Cast<LogicalComparisonJoin>().join_type == JoinType::SEMI) {
			ColumnBindingReplacer replacer;
			if (TryFoldSemiJoin(child, replacer)) {
				// Fix all column refs in the current node (and, recursively, its
				// children – which now include the newly-placed HNSW GET).
				replacer.VisitOperator(*plan);
				// child is now the HNSW GET (a leaf), nothing to recurse into.
				continue;
			}
		}
		// Not a SEMI JOIN we could fold – recurse normally
		FoldLateMaterializationJoins(child);
	}
}

//-----------------------------------------------------------------------------
// Redundant ORDER_BY elimination
//-----------------------------------------------------------------------------
// After FoldLateMaterializationJoins the plan may look like:
//
//   ORDER_BY(array_distance(vec, q))
//     PROJECTION(...)
//       HNSW_INDEX_SCAN(vec, id)
//
// HNSW_INDEX_SCAN already returns rows sorted by distance (ascending), so the
// ORDER_BY is redundant.  Remove it to avoid re-evaluating the distance
// expression on every returned row.
//
// We only remove ORDER_BY nodes whose entire subtree (modulo projections) is a
// single HNSW_INDEX_SCAN — any other data source could break order guarantees.
//-----------------------------------------------------------------------------

// Returns true iff the subtree rooted at `op` is an HNSW_INDEX_SCAN
// reachable through zero or more LOGICAL_PROJECTION nodes.
static bool SubtreeIsHNSWOnly(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		return op.Cast<LogicalGet>().function.name == "hnsw_index_scan";
	}
	if (op.type == LogicalOperatorType::LOGICAL_PROJECTION && op.children.size() == 1) {
		return SubtreeIsHNSWOnly(*op.children[0]);
	}
	return false;
}

// Walk the plan and replace any LOGICAL_ORDER_BY whose subtree is HNSW-only
// with that subtree directly (promoting the child past the sort).
static void EliminateRedundantOrders(unique_ptr<LogicalOperator> &plan) {
	if (plan->type == LogicalOperatorType::LOGICAL_ORDER_BY && plan->children.size() == 1 &&
	    SubtreeIsHNSWOnly(*plan->children[0])) {
		// Replace the ORDER_BY with its child; then recurse in case of nesting.
		plan = std::move(plan->children[0]);
		EliminateRedundantOrders(plan);
		return;
	}
	for (auto &child : plan->children) {
		EliminateRedundantOrders(child);
	}
}

//-----------------------------------------------------------------------------
// Register
//-----------------------------------------------------------------------------
void HNSWModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	db.config.optimizer_extensions.push_back(HNSWIndexScanOptimizer());
}

} // namespace duckdb
