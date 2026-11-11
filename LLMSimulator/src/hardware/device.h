#pragma once
#include <map>
#include <memory>
#include <set>
#include <list>
#include <unordered_map>
#include <unordered_set>

#include "common/type.h"
#include "dram/dram_type.h"
#include "dram/memory_config.h"
#include "hardware/executor.h"
#include "hardware/hardware_config.h"
#include "model/model_config.h"
#include "module/status.h"
#include "scheduler/sequence.h"
#include "dram/power.h"
#include "prefetch/ddr5_transfer_engine.h"

namespace llm_system {

class PrefetchCandidateQueue;
struct PrefetchCandidate;

struct InflightExpertEntry {
  uint64_t transfer_id = 0;
  TransferClass transfer_class = TransferClass::DEMAND;
  double dispatch_time_ns = 0.0;
  int layer_id = -1;
  int expert_id = -1;
  int source_stage = -1;
  int request_id = 0;
  int seq_id = 0;
  int token_id = 0;

  // Phase 4D.4: Pin intent metadata
  bool has_pin_intent = false;
  int pin_target_stage_id = -1;
  int pin_expiration_stage_id = -1;
  int prediction_rank = 0;
  double prediction_score = 0.0;
  uint64_t candidate_id = 0;
};

struct PrefetchPipelineMetrics {
  uint64_t candidates_evaluated = 0;
  uint64_t candidates_dispatched = 0;
  uint64_t prefetch_transfers_submitted = 0;
  uint64_t prefetch_transfers_completed = 0;
  uint64_t prefetches_inserted_hbm = 0;
  uint64_t prefetch_useful_hits = 0;
  uint64_t prefetch_promotions_to_demand = 0;
  uint64_t duplicate_prefetches_prevented = 0;
  uint64_t resident_prefetches_skipped = 0;
  uint64_t demand_inflight_attachments = 0;
  uint64_t cancelled_prefetches = 0;

  // Phase 2C Effectiveness Metrics
  uint64_t timely_resident_hits = 0;
  uint64_t late_inflight_attachments = 0;
  uint64_t not_prefetched_misses = 0;
  uint64_t demanded_transfers = 0;
  uint64_t unused_transfers = 0;
  uint64_t evicted_before_demand_transfers = 0;
  uint64_t still_inflight_at_end_transfers = 0;
};

// Phase 2C: Dispatched speculative transfer and demand outcomes
enum class SpeculativeTransferOutcome {
  UNDECIDED,
  DEMANDED,
  UNUSED,
  CANCELLED,
  EVICTED_BEFORE_DEMAND,
  STILL_IN_FLIGHT_AT_END
};

enum class DemandOutcome {
  TIMELY_RESIDENT,
  LATE_IN_FLIGHT,
  NOT_PREFETCHED
};

struct SpeculativeTransferRecord {
  uint64_t transfer_id = 0;
  uint64_t candidate_id = 0;
  int request_id = 0;
  int seq_id = 0;
  int token_id = 0;
  int source_stage = -1;
  int target_stage = -1;
  int expert_id = -1;
  double predict_time_ns = 0.0;
  double dispatch_time_ns = 0.0;
  double transfer_start_ns = 0.0;
  double transfer_complete_ns = 0.0;
  double actual_demand_ns = 0.0;
  double remaining_transfer_at_demand_ns = 0.0;
  double demand_arrival_to_ready_ns = 0.0;
  SpeculativeTransferOutcome outcome = SpeculativeTransferOutcome::UNDECIDED;
};

struct DemandOutcomeRecord {
  int request_id = 0;
  int seq_id = 0;
  int token_id = 0;
  int layer_id = 0;
  int expert_id = 0;
  double demand_time_ns = 0.0;
  double remaining_transfer_ns = 0.0;
  double demand_arrival_to_ready_ns = 0.0;
  DemandOutcome outcome = DemandOutcome::NOT_PREFETCHED;
};

// Phase 4D.3: PABI (Predictor-Aware Buffer Insertion) Types
enum class ExpertClassification {
  DEMAND_VERIFIED,
  SPECULATIVE
};

struct PinIntent {
  int layer_id = -1;
  int expert_id = -1;
  int source_stage = -1;
  int target_stage_id = -1;
  int expiration_stage_id = -1;
  int prediction_rank = 0;
  double prediction_score = 0.0;
  uint64_t candidate_id = 0;
  uint64_t transfer_id = 0;
  bool is_active = true;
};

struct ResidentExpertEntry {
  int layer_id = -1;
  int expert_id = -1;
  long long size_bytes = 0;
  ExpertClassification classification = ExpertClassification::DEMAND_VERIFIED;
  std::list<std::tuple<int, int, long long>>::iterator list_it;

  // Phase 4D.4: Resident Pin Metadata
  bool is_pinned = false;
  int pin_target_stage_id = -1;
  int pin_expiration_stage_id = -1;
  int prediction_rank = 0;
  double prediction_score = 0.0;
  uint64_t candidate_id = 0;

  // Phase 2C: Prefetch origin metadata
  bool is_prefetched = false;
  uint64_t transfer_id = 0;
  int source_stage = -1;
  int request_id = 0;
  int seq_id = 0;
  int token_id = 0;
  double dispatch_time_ns = 0.0;
  double completion_time_ns = 0.0;
  bool was_demanded = false;

  // Point 4: FA-LRU metadata
  uint32_t access_frequency = 1;
  uint32_t last_epoch = 0;

  // Implicit conversion to std::list iterator for seamless backward compatibility
  operator std::list<std::tuple<int, int, long long>>::iterator() const {
    return list_it;
  }
};

struct PabiCacheMetrics {
  uint64_t demand_insertions = 0;
  uint64_t speculative_insertions = 0;
  uint64_t speculative_promotions = 0;
  uint64_t speculative_evictions = 0;
  uint64_t demand_evictions = 0;
  uint64_t speculative_insertions_rejected = 0;
  uint64_t useful_prefetch_hits = 0;

  // Phase 4D.4: Pinning Metrics
  uint64_t pin_candidates_considered = 0;
  uint64_t pin_candidates_selected = 0;
  uint64_t pin_intents_created = 0;
  uint64_t resident_experts_pinned = 0;
  uint64_t inflight_pin_intents = 0;
  uint64_t pin_expirations = 0;
  uint64_t pins_cleared_on_promotion = 0;
  uint64_t pinned_eviction_skipped = 0;
  uint64_t speculative_reject_pinned_capacity = 0;
  uint64_t pinned_expert_demand_hits = 0;
  uint64_t unpinned_speculative_evictions = 0;
};

class Device : public std::enable_shared_from_this<Device> {
  friend class Executor;
  friend class Cluster;

 public:
  using Ptr = std::shared_ptr<Device>;

  [[nodiscard]] static Ptr Create(SystemConfig config, int device_total_rank,
                                  Cluster_ptr cluster) {
    Device::Ptr ptr = Ptr(new Device(config, device_total_rank, cluster));
    ptr->connectTopModuleGraph();
    return ptr;
  }

  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;
  hw_metric memory_capacity;

  SystemConfig config;

  // rank in node
  int device_local_rank;
  // rank in cluster
  int device_total_rank;

  Device() = default;

  Device::Ptr get_ptr() { return shared_from_this(); }

  TopModuleGraph_ptr top_module_graph;

  void connectTopModuleGraph();
  void reset_status();
  void reset_timeboard();
  void set_dependency();

  bool check_module_graph_remain();
  void add_module(std::string name, Module_ptr module);

  void setModelConfig(ModelConfig& _model_config) {
    model_config = _model_config;
  }

  time_ns get_time() { return status.device_time; }
  void set_time(time_ns time) { status.device_time = time; }

  // run with module_graph
  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);

  void restartGraph();

  void setPerformExecution(bool perform) { perform_execution = perform; }

  // execute operations and update time;
  void execution(Tensor_Ptr input, Tensor_Ptr weight, Tensor_Ptr output);
  void execution(LayerType layer_type,
                 const std::vector<Tensor_Ptr>& tensor_list,
                 const BatchedSequence::Ptr sequences_metadata,
                 const LayerInfo layer_info);

  // allocate DataObject;
  void setMemoryObject(Tensor_Ptr tensor);

  void run_ramulator(DRAMRequest_Ptr dram_request, MemoryTarget target = MemoryTarget::HBM);
  void run_ramulator_batch(const std::list<DRAMRequest_Ptr>& requests, MemoryTarget target = MemoryTarget::HBM);
  void run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor);

  void addExecutionCache(ExecStatus& exec_status, CacheKey key);

  void addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                         ProcessorType processor_type,
                         DRAMRequestType dram_reqeust_type, long size, MemoryTarget target = MemoryTarget::HBM);

  bool checkExecutionCache(ExecStatus& exec_status, CacheKey key);
  bool checkExecutionCache(CacheKey key);


  void setExecStatus(ExecStatus& exec_status_);

  ExecStatus getExecStatus();
  ExecStatus getHighExecStatus();
  ExecStatus getLowExecStatus();

  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);

  Cluster_ptr cluster;
  DRAMInterface_Ptr dram_interface;
  DRAMInterface_Ptr offchip_dram_interface;

  StatusBoard status;
  ModelConfig model_config;

  MMapController_Ptr mmap_controller;
  MMapController_Ptr offchip_mmap_controller;

  // Phase 5/DS-3 Expert Residency Tracker (LRU Cache)
  struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;  
    }
  };
  
  // Eviction policy & PABI settings (Phase 4D.3)
  EvictionPolicy eviction_policy = EvictionPolicy::LRU;
  bool pabi_quarantine_enabled = true;
  long long cache_capacity_bytes = 80000000000LL;
  long long speculative_buffer_capacity_bytes = 16000000000LL;
  long long resident_bytes = 0;
  long long resident_demand_bytes = 0;
  long long resident_speculative_bytes = 0;

  // Recency lists:
  // In LRU mode: lru_expert_list is the single recency list.
  // In PABI mode: demand_lru_list holds demand-verified experts; speculative_lru_list holds quarantined experts.
  std::list<std::tuple<int, int, long long>> demand_lru_list;
  std::list<std::tuple<int, int, long long>> speculative_lru_list;
  std::list<std::tuple<int, int, long long>> lru_expert_list;
  std::unordered_map<std::pair<int, int>, ResidentExpertEntry, PairHash> resident_experts;

  // Phase 4D.1 / 4D.2: Event-driven DDR5 transfer engine (per-device instance)
  std::shared_ptr<DDR5TransferEngine> ddr5_transfer_engine = nullptr;
  std::unordered_map<std::pair<int, int>, uint64_t, PairHash> inflight_experts;
  std::unordered_map<std::pair<int, int>, InflightExpertEntry, PairHash> inflight_registry;
  std::unordered_set<std::pair<int, int>, PairHash> prefetched_resident_experts;
  PrefetchPipelineMetrics prefetch_metrics;
  PabiCacheMetrics pabi_metrics;

  // Phase 4D.4: Belády-Style Predictor-Score Pinning
  bool pinning_enabled = false;
  int max_pinned_experts = 4;
  std::string pin_selection_policy = "top_ranked";
  std::string pin_expiration_policy = "stage_deadline";
  int current_stage_id = 0;
  std::unordered_map<std::pair<int, int>, PinIntent, PairHash> active_pin_intents;

  // Point 4: FA-LRU state
  uint32_t global_epoch = 0;
  int fa_lru_window_size = 16;
  std::unordered_set<std::pair<int, int>, PairHash> previously_evicted_experts;
  uint64_t reloaded_experts_count = 0;

  struct FALRUDiagnostics {
    uint64_t total_eviction_decisions = 0;
    uint64_t divergent_decisions = 0;
    uint64_t protected_demand_experts_count = 0;
    uint64_t unconsumed_speculative_victims = 0;
    uint64_t fallback_beyond_window_count = 0;
    uint64_t total_scanned_candidates = 0;
    uint64_t max_scan_depth = 0;
    std::unordered_map<uint32_t, uint64_t> victim_frequency_hist;
    std::unordered_map<uint32_t, uint64_t> window_frequency_hist;
    std::unordered_set<std::pair<int, int>, PairHash> protected_experts_set;
    std::unordered_set<std::pair<int, int>, PairHash> protected_reused_set;
    std::unordered_set<std::pair<int, int>, PairHash> protected_unreused_evicted_set;
  } fa_lru_diag;

  void dump_fa_lru_diagnostics(const std::string& filepath);

  void set_current_stage(int stage) {
    if (stage == 0 && current_stage_id > 0) {
      global_epoch++;
    }
    current_stage_id = stage;
  }
  bool is_pinning_active() const;
  std::vector<PrefetchCandidate> select_pin_candidates_for_stage(
      int target_stage_id,
      const std::vector<PrefetchCandidate>& candidates);
  int expire_pins_for_stage(int target_stage_id);
  int cleanup_unexecuted_stage_pins(int stage_id);

  bool has_resident_expert(int layer_id, int expert_id) const;
  bool is_expert_resident(int layer_id, int expert_id, bool is_demand_access = true, bool is_primary_demand = true);
  void dump_prefetch_effectiveness(const std::string& filepath = "");

  std::unordered_map<uint64_t, SpeculativeTransferRecord> speculative_transfers;
  std::vector<DemandOutcomeRecord> demand_outcomes;

  void mark_expert_resident(int layer_id, int expert_id, long long expert_size,
                            ExpertClassification classification = ExpertClassification::DEMAND_VERIFIED,
                            bool is_admitting_prefetch = false);
  void mark_expert_speculative(int layer_id, int expert_id, long long expert_size);
  void mark_expert_demand(int layer_id, int expert_id, long long expert_size);
  bool promote_speculative_to_demand(int layer_id, int expert_id);
  bool verify_pabi_invariants() const;

  // Point 4: FA-LRU helper methods
  void materialize_lazy_decay(ResidentExpertEntry& entry);
  std::list<std::tuple<int, int, long long>>::iterator select_fa_lru_victim();

  uint64_t submit_demand_transfer(int layer_id, int expert_id, double current_time_ns,
                                  uint64_t bytes = DDR5TransferEngine::DEFAULT_EXPERT_BYTES,
                                  int request_id = 0, int seq_id = 0, int token_id = 0);
  uint64_t submit_prefetch_transfer(int layer_id, int expert_id, double current_time_ns,
                                    uint64_t bytes = DDR5TransferEngine::DEFAULT_EXPERT_BYTES,
                                    int priority = 0,
                                    int source_stage = -1,
                                    uint64_t candidate_id = 0,
                                    int request_id = 0, int seq_id = 0, int token_id = 0);
  bool promote_inflight_to_demand(int layer_id, int expert_id, double current_time_ns);
  void process_completed_prefetches(double current_device_time);
  int dispatch_prefetch_candidates(std::shared_ptr<PrefetchCandidateQueue> queue,
                                   double current_time_ns,
                                   int max_candidates = -1);
  bool wait_for_demand_transfer(uint64_t transfer_id);
  bool is_expert_inflight(int layer_id, int expert_id) const;
  void clear_inflight_expert(int layer_id, int expert_id);
  void reset_expert_cache() {
      lru_expert_list.clear();
      demand_lru_list.clear();
      speculative_lru_list.clear();
      resident_experts.clear();
      inflight_experts.clear();
      inflight_registry.clear();
      active_pin_intents.clear();
      prefetched_resident_experts.clear();
      previously_evicted_experts.clear();
      reloaded_experts_count = 0;
      global_epoch = 0;
      current_stage_id = 0;
      prefetch_metrics = PrefetchPipelineMetrics();
      pabi_metrics = PabiCacheMetrics();
      resident_bytes = 0;
      resident_demand_bytes = 0;
      resident_speculative_bytes = 0;
      speculative_transfers.clear();
      demand_outcomes.clear();
      fa_lru_diag = FALRUDiagnostics();
      if (ddr5_transfer_engine != nullptr) {
          ddr5_transfer_engine->reset();
      }
  }

  bool perform_execution;

 private:
  ExecStatus high_exec_status;
  ExecStatus low_exec_status;
  ExecStatus exec_status;

  bool use_ramulator;

  Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster);

  void execution_ramulator(LayerType layer_type,
                           std::vector<Tensor_Ptr> tensor_list);

  void execution_ideal(LayerType layer_type,
                       std::vector<Tensor_Ptr> tensor_list);
};
}  // namespace llm_system