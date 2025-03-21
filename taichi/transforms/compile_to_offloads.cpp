#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/pass.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/extension.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

namespace irpass {
namespace {

std::function<void(const std::string &)>
make_pass_printer(bool verbose, const std::string &kernel_name, IRNode *ir) {
  if (!verbose) {
    return [](const std::string &) {};
  }
  return [ir, kernel_name](const std::string &pass) {
    TI_INFO("[{}] {}:", kernel_name, pass);
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  };
}

}  // namespace

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, kernel->get_name(), ir);
  print("Initial IR");

  if (!verbose && config.print_preprocessed_ir && start_from_ast) {
    TI_INFO("[{}] {}:", kernel->get_name(), "Preprocessed IR");
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  }

  if (autodiff_mode == AutodiffMode::kReverse) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  if (start_from_ast) {
    irpass::frontend_type_check(ir);
    irpass::lower_ast(ir);
    print("Lowered");
  }

  irpass::compile_taichi_functions(ir, config);

  irpass::eliminate_immutable_local_vars(ir);
  print("Immutable local vars eliminated");

  irpass::type_check(ir, config);
  print("Typechecked");
  irpass::analysis::verify(ir);

  // TODO: strictly enforce bit vectorization for x86 cpu and CUDA now
  //       create a separate CompileConfig flag for the new pass
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda ||
      config.arch == Arch::amdgpu) {
    irpass::bit_loop_vectorize(ir);
    irpass::type_check(ir, config);
    print("Bit Loop Vectorized");
    irpass::analysis::verify(ir);
  }

  // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
  irpass::lower_matrix_ptr(ir);
  print("Matrix ptr lowered");

  irpass::full_simplify(
      ir, config,
      {false, /*autodiff_enabled*/ autodiff_mode != AutodiffMode::kNone});
  print("Simplified I");
  irpass::analysis::verify(ir);

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::analysis::gather_meshfor_relation_types(ir);
  }

  if (config.debug && autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    // Check whether the kernel obeys the autodiff limitation e.g., gloabl data
    // access rule
    // This check should be performed in the forward kernel i.e., autodiff_mode
    // == AutodiffMode::kCheckAutodiffValid
    irpass::demote_atomics(ir, config);
    irpass::differentiation_validation_check(ir, config, kernel->get_name());
    irpass::analysis::verify(ir);
  }

  if (config.real_matrix_scalarize) {
    irpass::scalarize(ir);

    // Remove redundant MatrixInitStmt inserted during scalarization
    irpass::die(ir);
    print("Scalarized");
  }

  if (autodiff_mode == AutodiffMode::kReverse ||
      autodiff_mode == AutodiffMode::kForward) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir, config);

    irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ true});
    irpass::auto_diff(ir, config, autodiff_mode, ad_use_stack);
    // TODO: Be carefull with the full_simplify when do high-order autodiff
    irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false});
    print("Gradient");
    irpass::analysis::verify(ir);
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir, config, {kernel->get_name()});
    print("Bound checked");
    irpass::analysis::verify(ir);
  }

  irpass::flag_access(ir);
  print("Access flagged I");
  irpass::analysis::verify(ir);

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false});
  print("Simplified II");
  irpass::analysis::verify(ir);

  irpass::offload(ir, config);
  print("Offloaded");
  irpass::analysis::verify(ir);

  // TODO: This pass may be redundant as cfg_optimization() is already called
  //  in full_simplify().
  if (config.opt_level > 0 && config.cfg_optimization) {
    irpass::cfg_optimization(ir, false, /*autodiff_enabled*/ false,
                             !config.real_matrix_scalarize);
    print("Optimized by CFG");
    irpass::analysis::verify(ir);
  }

  irpass::flag_access(ir);
  print("Access flagged II");

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false});
  print("Simplified III");
  irpass::analysis::verify(ir);
}

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, kernel->get_name(), ir);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.

  auto amgr = std::make_unique<AnalysisManager>();

  print("Start offload_to_executable");
  irpass::analysis::verify(ir);

  if (config.detect_read_only) {
    irpass::detect_read_only(ir);
    print("Detect read-only accesses");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted I");
  irpass::analysis::verify(ir);
  if (config.cache_loop_invariant_global_vars) {
    irpass::cache_loop_invariant_global_vars(ir, config);
    print("Cache loop-invariant global vars");
  }

  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::type_check(ir, config);
    print("Dense struct-for demoted");
    irpass::analysis::verify(ir);
  }

  if (config.make_cpu_multithreading_loop && arch_is_cpu(config.arch)) {
    irpass::make_cpu_multithreaded_range_for(ir, config);
    irpass::type_check(ir, config);
    print("Make CPU multithreaded range-for");
    irpass::analysis::verify(ir);
  }

  if (is_extension_supported(config.arch, Extension::mesh) &&
      config.demote_no_access_mesh_fors) {
    irpass::demote_no_access_mesh_fors(ir);
    irpass::type_check(ir, config);
    print("No-access mesh-for demoted");
    irpass::analysis::verify(ir);
  }

  if (make_thread_local) {
    irpass::make_thread_local(ir, config);
    print("Make thread local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::make_mesh_thread_local(ir, config, {kernel->get_name()});
    print("Make mesh thread local");
    if (config.make_mesh_block_local && config.arch == Arch::cuda) {
      irpass::make_mesh_block_local(ir, config, {kernel->get_name()});
      print("Make mesh block local");
      irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false});
      print("Simplified X");
    }
  }

  if (make_block_local) {
    irpass::make_block_local(ir, config, {kernel->get_name()});
    print("Make block local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::demote_mesh_statements(ir, config, {kernel->get_name()});
    print("Demote mesh statements");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted II");
  irpass::analysis::verify(ir);

  if (is_extension_supported(config.arch, Extension::quant) &&
      config.quant_opt_atomic_demotion) {
    irpass::analysis::gather_uniquely_accessed_bit_structs(ir, amgr.get());
  }

  irpass::remove_range_assumption(ir);
  print("Remove range assumption");

  irpass::remove_loop_unique(ir);
  print("Remove loop_unique");
  irpass::analysis::verify(ir);

  if (lower_global_access) {
    irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false});
    print("Simplified before lower access");
    irpass::lower_access(ir, config, {kernel->no_activate, true});
    print("Access lowered");
    irpass::analysis::verify(ir);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify(ir);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify(ir);
  }

  irpass::demote_operations(ir, config);
  print("Operations demoted");

  irpass::full_simplify(ir, config,
                        {lower_global_access, /*autodiff_enabled*/ false});
  print("Simplified IV");

  if (determine_ad_stack_size) {
    irpass::determine_ad_stack_size(ir, config);
    print("Autodiff stack size determined");
  }

  if (is_extension_supported(config.arch, Extension::quant)) {
    irpass::optimize_bit_struct_stores(ir, config, amgr.get());
    print("Bit struct stores optimized");
  }

  if (config.arch == Arch::cuda && config.half2_vectorization &&
      !get_custom_cuda_library_path().empty()) {
    irpass::vectorize_half2(ir);

    irpass::type_check(ir, config);

    irpass::full_simplify(ir, config,
                          {lower_global_access, /*autodiff_enabled*/ false});

    irpass::flag_access(ir);
    print("Half2 vectorized");
  }

  // Final field registration correctness & type checking
  irpass::type_check(ir, config);
  irpass::analysis::verify(ir);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local,
                           bool start_from_ast) {
  TI_AUTO_PROF;

  compile_to_offloads(ir, config, kernel, verbose, autodiff_mode, ad_use_stack,
                      start_from_ast);

  offload_to_executable(
      ir, config, kernel, verbose,
      /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse &&
          ad_use_stack,
      lower_global_access, make_thread_local, make_block_local);
}

void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      bool start_from_ast) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, func->get_name(), ir);
  print("Initial IR");

  if (autodiff_mode == AutodiffMode::kReverse) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  if (start_from_ast) {
    irpass::frontend_type_check(ir);
    irpass::lower_ast(ir);
    print("Lowered");
  }

  if (config.real_matrix_scalarize) {
    irpass::scalarize(ir);

    // Remove redundant MatrixInitStmt inserted during scalarization
    irpass::die(ir);
    print("Scalarized");
  }

  irpass::lower_access(ir, config, {{}, true});
  print("Access lowered");
  irpass::analysis::verify(ir);

  irpass::die(ir);
  print("DIE");
  irpass::analysis::verify(ir);

  irpass::flag_access(ir);
  print("Access flagged III");
  irpass::analysis::verify(ir);

  irpass::type_check(ir, config);
  print("Typechecked");

  irpass::demote_operations(ir, config);
  print("Operations demoted");

  irpass::full_simplify(ir, config,
                        {false, autodiff_mode != AutodiffMode::kNone});
  print("Simplified");
  irpass::analysis::verify(ir);
}

}  // namespace irpass

}  // namespace taichi::lang
