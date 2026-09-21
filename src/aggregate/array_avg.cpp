#include "aggregate/array_avg.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"

#include <cstring>

namespace duckdb {

struct ArrayAvgBindData : public FunctionData {
	ArrayAvgBindData(LogicalType child_type, idx_t array_size)
	    : child_type(std::move(child_type)), array_size(array_size) {
	}

	LogicalType child_type;
	idx_t array_size;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ArrayAvgBindData>(child_type, array_size);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ArrayAvgBindData>();
		return child_type == other.child_type && array_size == other.array_size;
	}
};

struct ArrayAvgState {
	uint64_t count;
	double *sum;

	void Initialize() {
		count = 0;
		sum = nullptr;
	}

	void Destroy() {
		if (sum) {
			delete[] sum;
			sum = nullptr;
		}
	}
};

static idx_t ArrayAvgStateSize(AggregateStateInput &input) {
	return sizeof(ArrayAvgState);
}

static void ArrayAvgInitialize(AggregateStateInput &input, data_ptr_t *states, idx_t count) {
	for (idx_t i = 0; i < count; i++) {
		auto state = reinterpret_cast<ArrayAvgState *>(states[i]);
		state->Initialize();
	}
}

static void ArrayAvgDestructor(Vector &state_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(sdata);
	auto states = UnifiedVectorFormat::GetData<ArrayAvgState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		auto state = states[sdata.sel->get_index(i)];
		state->Destroy();
	}
}

static void ArrayAvgUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {
	auto &input = inputs[0];
	auto &bind_data = aggr_input_data.bind_data->Cast<ArrayAvgBindData>();
	const idx_t array_size = bind_data.array_size;
	const auto &child_type = bind_data.child_type;

	UnifiedVectorFormat input_data;
	UnifiedVectorFormat state_data;
	input.ToUnifiedFormat(input_data);
	state_vector.ToUnifiedFormat(state_data);

	auto states = UnifiedVectorFormat::GetData<ArrayAvgState *>(state_data);
	auto &child_vector = ArrayVector::GetChild(input);
	const auto &child_validity = FlatVector::Validity(child_vector);

	if (child_type.id() == LogicalTypeId::FLOAT) {
		auto child_data = FlatVector::GetData<float>(child_vector);
		for (idx_t i = 0; i < count; i++) {
			const auto input_idx = input_data.sel->get_index(i);
			const auto state_idx = state_data.sel->get_index(i);
			auto state = states[state_idx];

			if (!input_data.validity.RowIsValid(input_idx)) {
				continue;
			}

			const idx_t offset = input_idx * array_size;
			if (!child_validity.CheckAllValid(offset + array_size, offset)) {
				throw InvalidInputException("array_avg: array elements cannot be NULL");
			}

			if (!state->sum) {
				state->sum = new double[array_size]();
			}

			const float *src = child_data + offset;
			for (idx_t j = 0; j < array_size; j++) {
				state->sum[j] += static_cast<double>(src[j]);
			}
			state->count++;
		}
	} else if (child_type.id() == LogicalTypeId::DOUBLE) {
		auto child_data = FlatVector::GetData<double>(child_vector);
		for (idx_t i = 0; i < count; i++) {
			const auto input_idx = input_data.sel->get_index(i);
			const auto state_idx = state_data.sel->get_index(i);
			auto state = states[state_idx];

			if (!input_data.validity.RowIsValid(input_idx)) {
				continue;
			}

			const idx_t offset = input_idx * array_size;
			if (!child_validity.CheckAllValid(offset + array_size, offset)) {
				throw InvalidInputException("array_avg: array elements cannot be NULL");
			}

			if (!state->sum) {
				state->sum = new double[array_size]();
			}

			const double *src = child_data + offset;
			for (idx_t j = 0; j < array_size; j++) {
				state->sum[j] += src[j];
			}
			state->count++;
		}
	}
}

static void ArrayAvgCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                            idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<ArrayAvgBindData>();
	const idx_t array_size = bind_data.array_size;

	UnifiedVectorFormat source_data;
	UnifiedVectorFormat target_data;
	source_vector.ToUnifiedFormat(source_data);
	target_vector.ToUnifiedFormat(target_data);

	auto source_states = UnifiedVectorFormat::GetData<ArrayAvgState *>(source_data);
	auto target_states = UnifiedVectorFormat::GetData<ArrayAvgState *>(target_data);

	for (idx_t i = 0; i < count; i++) {
		const auto src_idx = source_data.sel->get_index(i);
		const auto tgt_idx = target_data.sel->get_index(i);
		auto src = source_states[src_idx];
		auto tgt = target_states[tgt_idx];

		if (src->count == 0) {
			continue;
		}

		if (tgt->count == 0) {
			tgt->count = src->count;
			if (src->sum) {
				tgt->sum = new double[array_size];
				std::memcpy(tgt->sum, src->sum, array_size * sizeof(double));
			}
		} else {
			tgt->count += src->count;
			if (src->sum) {
				if (!tgt->sum) {
					tgt->sum = new double[array_size]();
				}
				for (idx_t j = 0; j < array_size; j++) {
					tgt->sum[j] += src->sum[j];
				}
			}
		}
	}
}

static void ArrayAvgFinalize(Vector &state_vector, AggregateFinalizeInputData &finalize_input_data, Vector &result,
                             idx_t count, idx_t offset) {
	auto &bind_data = finalize_input_data.bind_data->Cast<ArrayAvgBindData>();
	const idx_t array_size = bind_data.array_size;
	const auto &child_type = bind_data.child_type;

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(state_data);
	auto states = UnifiedVectorFormat::GetData<ArrayAvgState *>(state_data);

	auto &result_child = ArrayVector::GetChildMutable(result);

	if (child_type.id() == LogicalTypeId::FLOAT) {
		auto result_data = FlatVector::GetDataMutable<float>(result_child);
		for (idx_t i = 0; i < count; i++) {
			const auto state_idx = state_data.sel->get_index(i);
			const auto result_idx = offset + i;
			auto state = states[state_idx];

			if (state->count == 0 || !state->sum) {
				FlatVector::SetNull(result, result_idx, true);
				continue;
			}

			const idx_t res_offset = result_idx * array_size;
			float *dst = result_data + res_offset;
			const double inv_count = 1.0 / static_cast<double>(state->count);
			for (idx_t j = 0; j < array_size; j++) {
				dst[j] = static_cast<float>(state->sum[j] * inv_count);
			}
		}
	} else if (child_type.id() == LogicalTypeId::DOUBLE) {
		auto result_data = FlatVector::GetDataMutable<double>(result_child);
		for (idx_t i = 0; i < count; i++) {
			const auto state_idx = state_data.sel->get_index(i);
			const auto result_idx = offset + i;
			auto state = states[state_idx];

			if (state->count == 0 || !state->sum) {
				FlatVector::SetNull(result, result_idx, true);
				continue;
			}

			const idx_t res_offset = result_idx * array_size;
			double *dst = result_data + res_offset;
			const double inv_count = 1.0 / static_cast<double>(state->count);
			for (idx_t j = 0; j < array_size; j++) {
				dst[j] = state->sum[j] * inv_count;
			}
		}
	}
}

static unique_ptr<FunctionData> ArrayAvgBind(BindAggregateFunctionInput &input) {
	auto &bound_function = input.GetBoundFunction();
	auto &arguments = input.GetArguments();
	if (arguments.empty()) {
		throw BinderException("array_avg requires at least one argument");
	}
	const auto &input_type = arguments[0]->GetReturnType();
	if (input_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("array_avg requires an ARRAY argument, got %s", input_type.ToString());
	}

	auto child_type = ArrayType::GetChildType(input_type);
	auto array_size = ArrayType::GetSize(input_type);

	if (child_type.id() != LogicalTypeId::FLOAT && child_type.id() != LogicalTypeId::DOUBLE) {
		throw BinderException("array_avg only supports FLOAT and DOUBLE elements, got %s", child_type.ToString());
	}

	bound_function.GetArguments()[0] = input_type;
	bound_function.SetReturnType(input_type);

	return make_uniq<ArrayAvgBindData>(child_type, array_size);
}

void ArrayAvgFunction::Register(ExtensionLoader &loader) {
	AggregateFunction array_avg_fun("array_avg", {LogicalType::ANY}, LogicalType::ANY, ArrayAvgStateSize,
	                                ArrayAvgInitialize, ArrayAvgUpdate, ArrayAvgCombine, ArrayAvgFinalize, nullptr,
	                                ArrayAvgBind, ArrayAvgDestructor);
	AggregateFunctionSet array_avg_set("array_avg");
	array_avg_set.AddFunction(array_avg_fun);
	loader.RegisterFunction(array_avg_set);

	AggregateFunction vector_avg_fun("vector_avg", {LogicalType::ANY}, LogicalType::ANY, ArrayAvgStateSize,
	                                 ArrayAvgInitialize, ArrayAvgUpdate, ArrayAvgCombine, ArrayAvgFinalize, nullptr,
	                                 ArrayAvgBind, ArrayAvgDestructor);
	AggregateFunctionSet vector_avg_set("vector_avg");
	vector_avg_set.AddFunction(vector_avg_fun);
	loader.RegisterFunction(vector_avg_set);
}

} // namespace duckdb
