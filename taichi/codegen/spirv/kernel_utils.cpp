#include "taichi/codegen/spirv/kernel_utils.h"

#include <unordered_map>

#include "taichi/program/kernel.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {
namespace spirv {

// static
std::string TaskAttributes::buffers_name(BufferInfo b) {
  if (b.type == BufferType::Args) {
    return "Args";
  }
  if (b.type == BufferType::Rets) {
    return "Rets";
  }
  if (b.type == BufferType::GlobalTmps) {
    return "GlobalTmps";
  }
  if (b.type == BufferType::Root) {
    return std::string("Root: ") + std::to_string(b.root_id);
  }
  TI_ERROR("unrecognized buffer type");
}

std::string TaskAttributes::debug_string() const {
  std::string result;
  result += fmt::format(
      "<TaskAttributes name={} advisory_total_num_threads={} "
      "task_type={} buffers=[ ",
      name, advisory_total_num_threads, offloaded_task_type_name(task_type));
  for (auto b : buffer_binds) {
    result += buffers_name(b.buffer) + " ";
  }
  result += "]";  // closes |buffers|
  // TODO(k-ye): show range_for
  result += ">";
  return result;
}

std::string TaskAttributes::BufferBind::debug_string() const {
  return fmt::format("<type={} binding={}>",
                     TaskAttributes::buffers_name(buffer), binding);
}

size_t KernelContextAttributes::recursive_process_struct(
    const StructType& struct_type,
    std::vector<RetAttributes>& ret_attributes,
    size_t bytes) {
  for (const auto& member : struct_type.elements()) {
    if (auto* struct_type_ = member.type->cast<StructType>()) {
      bytes = recursive_process_struct(*struct_type_, ret_attributes, bytes);
    } else if (auto tensor_type = member.type->cast<TensorType>()) {
      auto tensor_dtype = tensor_type->get_element_type();
      TI_ASSERT(tensor_dtype->is<PrimitiveType>());
      RetAttributes ra;
      ra.dtype = tensor_dtype->cast<PrimitiveType>()->type;
      ra.is_array = true;
      size_t dt_bytes = data_type_size(tensor_dtype);
      ra.stride = tensor_type->get_num_elements() * dt_bytes;
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      ra.offset_in_mem = bytes;
      bytes += ra.stride;
      ra.index = ret_attributes.size();
      ret_attributes.push_back(ra);
    } else {
      TI_ASSERT(member.type->is<PrimitiveType>());
      RetAttributes ra;
      ra.dtype = member.type->cast<PrimitiveType>()->type;
      ra.is_array = false;
      size_t dt_bytes = data_type_size(member.type->cast<PrimitiveType>());
      ra.stride = dt_bytes;
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      ra.offset_in_mem = bytes;
      bytes += ra.stride;
      ra.index = ret_attributes.size();
      ret_attributes.push_back(ra);
    }
  }
  return bytes;
}

KernelContextAttributes::KernelContextAttributes(
    const Kernel &kernel,
    const DeviceCapabilityConfig *caps)
    : args_bytes_(0), rets_bytes_(0) {
  arr_access.resize(kernel.parameter_list.size(), irpass::ExternalPtrAccess(0));
  arg_attribs_vec_.reserve(kernel.parameter_list.size());
  // TODO: We should be able to limit Kernel args and rets to be primitive types
  // as well but let's leave that as a followup up PR.
  for (const auto &ka : kernel.parameter_list) {
    ArgAttributes aa;
    aa.name = ka.name;
    aa.is_array = ka.is_array;
    arg_attribs_vec_.push_back(aa);
  }
  size_t bytes = 0;
  for (const auto &kr : kernel.rets) {
    if (auto tensor_type = kr.dt->cast<TensorType>()) {
      auto tensor_dtype = tensor_type->get_element_type();
      TI_ASSERT(tensor_dtype->is<PrimitiveType>());
      RetAttributes ra;
      ra.dtype = tensor_dtype->cast<PrimitiveType>()->type;
      ra.is_array = true;
      size_t dt_bytes = data_type_size(tensor_dtype);
      ra.stride = tensor_type->get_num_elements() * dt_bytes;
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      ra.offset_in_mem = bytes;
      bytes += ra.stride;
      ra.index = ret_attribs_vec_.size();
      ret_attribs_vec_.push_back(ra);
    } else if (auto struct_type = kr.dt->cast<StructType>()) {
      bytes = recursive_process_struct(*struct_type, ret_attribs_vec_, bytes);
    } else {
      TI_ASSERT(kr.dt->is<PrimitiveType>());
      RetAttributes ra;
      ra.dtype = kr.dt->cast<PrimitiveType>()->type;
      ra.is_array = false;
      size_t dt_bytes = data_type_size(kr.dt);
      ra.stride = dt_bytes;
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      ra.offset_in_mem = bytes;
      bytes += ra.stride;
      ra.index = ret_attribs_vec_.size();
      ret_attribs_vec_.push_back(ra);
    }
  }
//  auto arange_args = [](auto *vec, size_t offset) -> size_t {
//    size_t bytes = offset;
//    for (int i = 0; i < vec->size(); ++i) {
//      auto &attribs = (*vec)[i];
//      const size_t dt_bytes = data_type_size(PrimitiveType::get(attribs.dtype));
//      // Align bytes to the nearest multiple of dt_bytes
//      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
//      attribs.offset_in_mem = bytes;
//      bytes += attribs.stride;
//      TI_TRACE(
//          "  at={} {} offset_in_mem={} stride={}",
//          (*vec)[i].is_array ? "array" : "scalar", i,
//          attribs.offset_in_mem, attribs.stride);
//    }
//    return bytes - offset;
//  };

  args_type_ = kernel.args_type;
  rets_type_ = kernel.ret_type;

  args_bytes_ = kernel.args_size;

//  TI_TRACE("rets:");
  rets_bytes_ = bytes;
//
//  TI_ASSERT(ret_attribs_vec_.size() == kernel.ret_type->elements().size());
//  for (int i = 0; i < ret_attribs_vec_.size(); ++i) {
//    TI_ASSERT(ret_attribs_vec_[i].offset_in_mem ==
//              kernel.ret_type->get_element_offset({i}));
//  }
//
//  TI_ASSERT(rets_bytes_ == kernel.ret_size);
//
//  TI_TRACE("sizes: args={} rets={}", args_bytes(), rets_bytes());
//  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

}  // namespace spirv
}  // namespace taichi::lang
