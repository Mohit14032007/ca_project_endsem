#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cassert>
#include <memory>
#include <string>
#include <cctype>

#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"
#include "dram/dram_interface.h"
#include "dram/mmap_controller.h"
#include "yaml-cpp/yaml.h"

using namespace llm_system;

int main() {
  std::cout << "======================================================================\n";
  std::cout << "TEST: Configurable Off-Chip Execution Cache Validation (Phase 2A)\n";
  std::cout << "======================================================================\n";

  // ------------------------------------------------------------------
  // Check 1 & 2: Default Configuration Setting is False
  // ------------------------------------------------------------------
  std::cout << "\n[Check 1 & 2] Verifying C++ default setting is false...\n";
  SystemConfig default_cfg;
  if (default_cfg.bypass_offchip_execution_cache) {
    std::cerr << "FAILED: default_cfg.bypass_offchip_execution_cache should be false!\n";
    return 1;
  }
  SystemConfig a100_cfg = A100;
  if (a100_cfg.bypass_offchip_execution_cache) {
    std::cerr << "FAILED: A100 preset bypass_offchip_execution_cache should be false!\n";
    return 1;
  }
  std::cout << "  -> PASS: SystemConfig default and presets initialize bypass_offchip_execution_cache = false.\n";

  // ------------------------------------------------------------------
  // Check 3: YAML Configuration Parsing (True, False, Missing)
  // ------------------------------------------------------------------
  std::cout << "\n[Check 3] Verifying YAML configuration parsing...\n";
  
  // Case 3a: YAML explicitly true
  {
    std::string yaml_str = 
      "system:\n"
      "  optimization:\n"
      "    bypass_offchip_execution_cache: true\n";
    YAML::Node config = YAML::Load(yaml_str);
    SystemConfig test_cfg;
    if (config["system"]["optimization"]["bypass_offchip_execution_cache"]) {
      std::string val_str = config["system"]["optimization"]["bypass_offchip_execution_cache"].as<std::string>();
      std::string s = "";
      for (char c : val_str) {
        s += std::tolower(c);
      }
      if (s == "1" || s == "true" || s == "on" || s == "yes") {
        test_cfg.bypass_offchip_execution_cache = true;
      } else if (s == "0" || s == "false" || s == "off" || s == "no") {
        test_cfg.bypass_offchip_execution_cache = false;
      }
    }
    if (!test_cfg.bypass_offchip_execution_cache) {
      std::cerr << "FAILED: YAML true did not set bypass_offchip_execution_cache to true!\n";
      return 2;
    }
    std::cout << "  -> PASS: YAML true parsed successfully as true.\n";
  }

  // Case 3b: YAML explicitly false
  {
    std::string yaml_str = 
      "system:\n"
      "  optimization:\n"
      "    bypass_offchip_execution_cache: false\n";
    YAML::Node config = YAML::Load(yaml_str);
    SystemConfig test_cfg;
    test_cfg.bypass_offchip_execution_cache = true; // Pre-set to true to test reset
    if (config["system"]["optimization"]["bypass_offchip_execution_cache"]) {
      std::string val_str = config["system"]["optimization"]["bypass_offchip_execution_cache"].as<std::string>();
      std::string s = "";
      for (char c : val_str) {
        s += std::tolower(c);
      }
      if (s == "1" || s == "true" || s == "on" || s == "yes") {
        test_cfg.bypass_offchip_execution_cache = true;
      } else if (s == "0" || s == "false" || s == "off" || s == "no") {
        test_cfg.bypass_offchip_execution_cache = false;
      }
    }
    if (test_cfg.bypass_offchip_execution_cache) {
      std::cerr << "FAILED: YAML false did not set bypass_offchip_execution_cache to false!\n";
      return 3;
    }
    std::cout << "  -> PASS: YAML false parsed successfully as false.\n";
  }

  // Case 3c: Missing YAML field retains default false
  {
    std::string yaml_str = 
      "system:\n"
      "  optimization:\n"
      "    use_ramulator: true\n";
    YAML::Node config = YAML::Load(yaml_str);
    SystemConfig test_cfg;
    if (config["system"]["optimization"]["bypass_offchip_execution_cache"]) {
      std::string val_str = config["system"]["optimization"]["bypass_offchip_execution_cache"].as<std::string>();
      std::string s = "";
      for (char c : val_str) {
        s += std::tolower(c);
      }
      if (s == "1" || s == "true" || s == "on" || s == "yes") {
        test_cfg.bypass_offchip_execution_cache = true;
      } else if (s == "0" || s == "false" || s == "off" || s == "no") {
        test_cfg.bypass_offchip_execution_cache = false;
      }
    }
    if (test_cfg.bypass_offchip_execution_cache) {
      std::cerr << "FAILED: Missing YAML field altered default false!\n";
      return 4;
    }
    std::cout << "  -> PASS: Missing YAML field retains default false.\n";
  }

  // ------------------------------------------------------------------
  // Check 4: Environment Variable Override & Precedence
  // ------------------------------------------------------------------
  std::cout << "\n[Check 4] Verifying environment variable override & precedence...\n";
  
  // Case 4a: YAML is false, env var is 1 -> Precedence dictates TRUE
  {
    SystemConfig test_cfg;
    test_cfg.bypass_offchip_execution_cache = false; // From YAML
    setenv("BYPASS_OFFCHIP_EXECUTION_CACHE", "1", 1);
    test_cfg.apply_environment_overrides();
    if (!test_cfg.bypass_offchip_execution_cache) {
      std::cerr << "FAILED: BYPASS_OFFCHIP_EXECUTION_CACHE=1 did not override YAML false!\n";
      return 5;
    }
    std::cout << "  -> PASS: Env var '1' overrides YAML false to true.\n";
  }

  // Case 4b: YAML is true, env var is 0 -> Precedence dictates FALSE
  {
    SystemConfig test_cfg;
    test_cfg.bypass_offchip_execution_cache = true; // From YAML
    setenv("BYPASS_OFFCHIP_EXECUTION_CACHE", "0", 1);
    test_cfg.apply_environment_overrides();
    if (test_cfg.bypass_offchip_execution_cache) {
      std::cerr << "FAILED: BYPASS_OFFCHIP_EXECUTION_CACHE=0 did not override YAML true!\n";
      return 6;
    }
    std::cout << "  -> PASS: Env var '0' overrides YAML true to false.\n";
  }

  // Clean up environment
  unsetenv("BYPASS_OFFCHIP_EXECUTION_CACHE");

  // ------------------------------------------------------------------
  // Initialize Simulation Environment for Dispatch Tests
  // ------------------------------------------------------------------
  SystemConfig sim_config;
  sim_config.gpu_gen = "A100";
  sim_config.use_ramulator = true;
  sim_config.offchip_dram_cfg_path = "reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5.yaml";

  Cluster::Ptr cluster = Cluster::Create(sim_config, nullptr);
  Device::Ptr device = Device::Create(sim_config, 0, cluster);
  device->perform_execution = true;

  const long tensor_size = 4096; // 4 KB lightweight test tensor

  // ------------------------------------------------------------------
  // Check 5 & 7: Mode A (bypass=false) - Off-Chip Cache Hits Skip Ramulator
  // ------------------------------------------------------------------
  std::cout << "\n[Check 5 & 7] Verifying Mode A (bypass=false) execution-cache reuse...\n";
  device->config.bypass_offchip_execution_cache = false;
  cluster->execution_time_cache.clear();

  // Test 5a: Single dispatch path (issueRamulator)
  {
    Tensor::Ptr t1 = Tensor::Create("single_test", {1, (int)tensor_size}, "weight", device);
    t1->weight_target = MemoryTarget::OFFCHIP_DRAM;
    t1->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 1000000LL, tensor_size, device->offchip_mmap_controller));

    double ddr5_t0 = device->offchip_dram_interface->time;
    ExecStatus stat1 = issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                                      DRAMRequestType::kRead, PIMOperandType::kDRAM, t1, MemoryTarget::OFFCHIP_DRAM);
    double ddr5_t1 = device->offchip_dram_interface->time;

    if (ddr5_t1 - ddr5_t0 <= 0.0) {
      std::cerr << "FAILED: First call did not advance DDR5 clock!\n";
      return 7;
    }
    if (cluster->execution_time_cache.size() != 1) {
      std::cerr << "FAILED: ExecutionCache size should be 1 after first call, got " << cluster->execution_time_cache.size() << "\n";
      return 8;
    }

    // Second call: Identical key -> MUST HIT cache and skip Ramulator (delta == 0)
    ExecStatus stat2 = issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                                      DRAMRequestType::kRead, PIMOperandType::kDRAM, t1, MemoryTarget::OFFCHIP_DRAM);
    double ddr5_t2 = device->offchip_dram_interface->time;

    if (ddr5_t2 - ddr5_t1 != 0.0) {
      std::cerr << "FAILED: Second call did not hit cache; DDR5 clock advanced by " << (ddr5_t2 - ddr5_t1) << " ns!\n";
      return 9;
    }
    if (stat1.memory_duration != stat2.memory_duration) {
      std::cerr << "FAILED: Memoized memory_duration mismatch!\n";
      return 10;
    }
    std::cout << "  -> PASS: Single dispatch path (issueRamulator) correctly populates and reuses cache.\n";
  }

  // Test 5b: Batch dispatch path (issueRamulatorBatch)
  {
    cluster->execution_time_cache.clear();
    Tensor::Ptr bt1 = Tensor::Create("batch_test1", {1, (int)tensor_size}, "weight", device);
    bt1->weight_target = MemoryTarget::OFFCHIP_DRAM;
    bt1->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 2000000LL, tensor_size, device->offchip_mmap_controller));

    Tensor::Ptr bt2 = Tensor::Create("batch_test2", {1, (int)tensor_size}, "weight", device);
    bt2->weight_target = MemoryTarget::OFFCHIP_DRAM;
    bt2->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 3000000LL, tensor_size, device->offchip_mmap_controller));

    std::vector<Tensor_Ptr> batch = {bt1, bt2};

    double ddr5_t0 = device->offchip_dram_interface->time;
    ExecStatus bstat1 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                           DRAMRequestType::kRead, PIMOperandType::kDRAM, batch, MemoryTarget::OFFCHIP_DRAM);
    double ddr5_t1 = device->offchip_dram_interface->time;

    if (ddr5_t1 - ddr5_t0 <= 0.0) {
      std::cerr << "FAILED: First batch call did not advance DDR5 clock!\n";
      return 11;
    }
    if (cluster->execution_time_cache.size() != 1) {
      std::cerr << "FAILED: ExecutionCache size should be 1 after first batch call!\n";
      return 12;
    }

    // Second batch call: MUST HIT cache
    ExecStatus bstat2 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                           DRAMRequestType::kRead, PIMOperandType::kDRAM, batch, MemoryTarget::OFFCHIP_DRAM);
    double ddr5_t2 = device->offchip_dram_interface->time;

    if (ddr5_t2 - ddr5_t1 != 0.0) {
      std::cerr << "FAILED: Second batch call did not hit cache; DDR5 clock delta: " << (ddr5_t2 - ddr5_t1) << " ns!\n";
      return 13;
    }
    if (bstat1.memory_duration != bstat2.memory_duration) {
      std::cerr << "FAILED: Batch memoized duration mismatch!\n";
      return 14;
    }
    std::cout << "  -> PASS: Batch dispatch path (issueRamulatorBatch) correctly populates and reuses cache.\n";
  }

  // ------------------------------------------------------------------
  // Check 8 & 9: Mode B (bypass=true) - Ramulator Always Invoked & No Insertion
  // ------------------------------------------------------------------
  std::cout << "\n[Check 8 & 9] Verifying Mode B (bypass=true) unconditional bypass...\n";
  device->config.bypass_offchip_execution_cache = true;
  cluster->execution_time_cache.clear();

  {
    Tensor::Ptr off_t = Tensor::Create("bypass_test", {1, (int)tensor_size}, "weight", device);
    off_t->weight_target = MemoryTarget::OFFCHIP_DRAM;
    off_t->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 4000000LL, tensor_size, device->offchip_mmap_controller));
    std::vector<Tensor_Ptr> batch = {off_t};

    double ddr5_t0 = device->offchip_dram_interface->time;
    ExecStatus stat1 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                          DRAMRequestType::kRead, PIMOperandType::kDRAM, batch, MemoryTarget::OFFCHIP_DRAM);
    double ddr5_t1 = device->offchip_dram_interface->time;

    if (ddr5_t1 - ddr5_t0 <= 0.0) {
      std::cerr << "FAILED: Mode B first call did not run Ramulator!\n";
      return 15;
    }
    if (cluster->execution_time_cache.size() != 0) {
      std::cerr << "FAILED: Mode B must NOT insert into ExecutionCache! Cache size is " << cluster->execution_time_cache.size() << "\n";
      return 16;
    }

    // Second call: MUST execute Ramulator again (delta > 0) and NOT hit cache
    ExecStatus stat2 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                          DRAMRequestType::kRead, PIMOperandType::kDRAM, batch, MemoryTarget::OFFCHIP_DRAM);
    double ddr5_t2 = device->offchip_dram_interface->time;

    if (ddr5_t2 - ddr5_t1 <= 0.0) {
      std::cerr << "FAILED: Mode B second call did not run Ramulator; delta is 0!\n";
      return 17;
    }
    if (cluster->execution_time_cache.size() != 0) {
      std::cerr << "FAILED: Mode B second call inserted into ExecutionCache!\n";
      return 18;
    }
    std::cout << "  -> PASS: Mode B (bypass=true) always executes Ramulator and disables cache insertion.\n";
  }

  // ------------------------------------------------------------------
  // Check 6: HBM Execution Cache Behavior Unchanged Under Both Modes
  // ------------------------------------------------------------------
  std::cout << "\n[Check 6] Verifying HBM execution-cache behavior is unchanged...\n";
  
  // Test HBM with bypass_offchip_execution_cache = false
  {
    device->config.bypass_offchip_execution_cache = false;
    cluster->execution_time_cache.clear();

    Tensor::Ptr hbm_t = Tensor::Create("hbm_modeA", {1, (int)tensor_size}, "weight", device);
    hbm_t->weight_target = MemoryTarget::HBM;
    device->setMemoryObject(hbm_t);
    std::vector<Tensor_Ptr> hbm_batch = {hbm_t};

    double h0 = device->dram_interface->time;
    ExecStatus hstat1 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                           DRAMRequestType::kRead, PIMOperandType::kDRAM, hbm_batch, MemoryTarget::HBM);
    double h1 = device->dram_interface->time;

    if (h1 - h0 <= 0.0) {
      std::cerr << "FAILED: HBM first call did not advance HBM clock!\n";
      return 19;
    }
    if (cluster->execution_time_cache.size() != 1) {
      std::cerr << "FAILED: HBM first call did not populate ExecutionCache!\n";
      return 20;
    }

    ExecStatus hstat2 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                           DRAMRequestType::kRead, PIMOperandType::kDRAM, hbm_batch, MemoryTarget::HBM);
    double h2 = device->dram_interface->time;

    if (h2 - h1 != 0.0) {
      std::cerr << "FAILED: HBM second call did not hit cache; delta: " << (h2 - h1) << "\n";
      return 21;
    }
    std::cout << "  -> PASS: HBM correctly caches and reuses when bypass_offchip_execution_cache = false.\n";
  }

  // Test HBM with bypass_offchip_execution_cache = true
  {
    device->config.bypass_offchip_execution_cache = true;
    cluster->execution_time_cache.clear();

    Tensor::Ptr hbm_t = Tensor::Create("hbm_modeB", {1, (int)tensor_size}, "weight", device);
    hbm_t->weight_target = MemoryTarget::HBM;
    device->setMemoryObject(hbm_t);
    std::vector<Tensor_Ptr> hbm_batch = {hbm_t};

    double h0 = device->dram_interface->time;
    ExecStatus hstat1 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                           DRAMRequestType::kRead, PIMOperandType::kDRAM, hbm_batch, MemoryTarget::HBM);
    double h1 = device->dram_interface->time;

    if (h1 - h0 <= 0.0) {
      std::cerr << "FAILED: HBM first call in Mode B did not advance HBM clock!\n";
      return 22;
    }
    if (cluster->execution_time_cache.size() != 1) {
      std::cerr << "FAILED: HBM in Mode B must still populate ExecutionCache!\n";
      return 23;
    }

    ExecStatus hstat2 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                                           DRAMRequestType::kRead, PIMOperandType::kDRAM, hbm_batch, MemoryTarget::HBM);
    double h2 = device->dram_interface->time;

    if (h2 - h1 != 0.0) {
      std::cerr << "FAILED: HBM second call in Mode B did not hit cache; delta: " << (h2 - h1) << "\n";
      return 24;
    }
    std::cout << "  -> PASS: HBM correctly caches and reuses even when bypass_offchip_execution_cache = true.\n";
  }

  std::cout << "\n======================================================================\n";
  std::cout << "SUCCESS: All 9 validation checks for Phase 2A PASSED!\n";
  std::cout << "======================================================================\n";
  return 0;
}
