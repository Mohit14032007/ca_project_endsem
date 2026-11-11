#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm_system {

/**
 * Transfer class denoting priority level.
 * DEMAND: Highest priority, FIFO service, preempts active prefetch.
 * PREFETCH: Speculative priority, serviced only when no demand is queued or active.
 */
enum class TransferClass {
    DEMAND,
    PREFETCH
};

std::string to_string(TransferClass tc);

/**
 * Lifecycle states of an off-chip DDR5 memory transfer.
 * Explicitly distinguished from Phase 3 prediction candidate states.
 */
enum class TransferState {
    QUEUED,     // Waiting in demand or prefetch queue
    ACTIVE,     // Transmitting on the DDR5 channel
    PAUSED,     // Preempted by demand; progress preserved
    COMPLETED,  // All bytes transferred successfully
    CANCELLED   // Cancelled before completion
};

std::string to_string(TransferState ts);

/**
 * Event types handled by the discrete-event simulation engine.
 */
enum class TransferEventType {
    COMPLETION,
    DEMAND_ARRIVAL,
    PREFETCH_ARRIVAL
};

std::string to_string(TransferEventType tet);

/**
 * Memory transfer representation tracking byte-level progress and timing.
 */
struct MemoryTransfer {
    uint64_t transfer_id = 0;
    int request_id = -1;
    int seq_id = -1;
    int token_id = -1;
    int layer_id = -1;
    int expert_id = -1;
    uint64_t candidate_id = 0;
    TransferClass transfer_class = TransferClass::DEMAND;

    uint64_t total_bytes = 88080384ULL;       // Calibrated DeepSeek-V3 expert footprint
    double remaining_bytes = 88080384.0;
    double transferred_bytes = 0.0;

    double arrival_time_ns = 0.0;
    double service_start_time_ns = -1.0;      // First time state transitioned to ACTIVE
    double last_start_time_ns = -1.0;         // Most recent time state transitioned to ACTIVE
    double completion_time_ns = -1.0;

    TransferState state = TransferState::QUEUED;
    uint64_t generation = 0;                  // Stale-event protection counter
    int priority = 0;                         // Higher value indicates higher scheduling priority

    int num_tensors = 3;
    uint64_t tensor_size = 29360128ULL;       // Single projection tensor footprint
    int completed_tensors = 0;

    bool is_completed() const {
        if (state == TransferState::COMPLETED) {
            return true;
        }
        return false;
    }

    bool is_active() const {
        if (state == TransferState::ACTIVE) {
            return true;
        }
        return false;
    }

    bool is_paused() const {
        if (state == TransferState::PAUSED) {
            return true;
        }
        return false;
    }

    bool is_queued() const {
        if (state == TransferState::QUEUED) {
            return true;
        }
        return false;
    }
};

/**
 * Discrete event scheduled in the event engine's priority queue.
 */
struct TransferEvent {
    uint64_t event_id = 0;
    double timestamp_ns = 0.0;
    TransferEventType type = TransferEventType::COMPLETION;
    uint64_t transfer_id = 0;
    uint64_t generation = 0;

    // Strict deterministic ordering:
    // 1. Ascending by timestamp_ns.
    // 2. COMPLETION before ARRIVAL on exact timestamp collision.
    // 3. Ascending by event_id for tie-breaking identical types.
    bool operator>(const TransferEvent& other) const {
        if (timestamp_ns != other.timestamp_ns) {
            if (timestamp_ns > other.timestamp_ns) {
                return true;
            }
            return false;
        }

        // Exact timestamp collision tie-break: COMPLETION has priority over ARRIVAL
        int p_self = 0;
        if (type == TransferEventType::COMPLETION) {
            p_self = 0;
        } else if (type == TransferEventType::DEMAND_ARRIVAL) {
            p_self = 1;
        } else {
            p_self = 2;
        }

        int p_other = 0;
        if (other.type == TransferEventType::COMPLETION) {
            p_other = 0;
        } else if (other.type == TransferEventType::DEMAND_ARRIVAL) {
            p_other = 1;
        } else {
            p_other = 2;
        }

        if (p_self != p_other) {
            if (p_self > p_other) {
                return true;
            }
            return false;
        }

        if (event_id > other.event_id) {
            return true;
        }
        return false;
    }
};

/**
 * Metric counters and accumulators supporting Phase 4C/4D/4E evaluation.
 */
struct TransferEngineMetrics {
    uint64_t total_demand_submitted = 0;
    uint64_t total_prefetch_submitted = 0;
    uint64_t demand_completions = 0;
    uint64_t prefetch_completions = 0;
    uint64_t prefetch_preemptions = 0;
    uint64_t prefetch_resumptions = 0;
    uint64_t total_bytes_transferred = 0;
    double demand_service_time_ns = 0.0;
    double prefetch_service_time_ns = 0.0;
    double demand_queue_waiting_time_ns = 0.0;
    double prefetch_queue_waiting_time_ns = 0.0;
    uint64_t stale_events_ignored = 0;
    uint64_t cancelled_transfers = 0;
};

/**
 * Calibrated DDR5 Asynchronous Transfer Engine.
 * Implements a single-server serialized channel calibrated from Ramulator2.
 * Strictly zero ternary operators across all members.
 */
class DDR5TransferEngine {
public:
    // Calibrated constants from Phase 4B / 4B.1
    // 22.578e9 bytes/sec = 22.578 bytes/nanosecond
    static constexpr double DEFAULT_BANDWIDTH_BYTES_PER_NS = 22.578;
    static constexpr uint64_t DEFAULT_EXPERT_BYTES = 88080384ULL;
    static constexpr uint64_t DEFAULT_TENSOR_BYTES = 29360128ULL;
    static constexpr int DEFAULT_TENSORS_PER_EXPERT = 3;

    explicit DDR5TransferEngine(double bandwidth_bytes_per_ns = DEFAULT_BANDWIDTH_BYTES_PER_NS);
    ~DDR5TransferEngine() = default;

    // Submission interfaces
    bool submit_transfer(const MemoryTransfer& transfer, double current_time_ns);
    bool submit_demand(int request_id, int layer_id, int expert_id, double current_time_ns,
                       uint64_t bytes = DEFAULT_EXPERT_BYTES, uint64_t transfer_id = 0,
                       int seq_id = 0, int token_id = 0);
    uint64_t submit_demand_id(int request_id, int layer_id, int expert_id, double current_time_ns,
                              uint64_t bytes = DEFAULT_EXPERT_BYTES, uint64_t transfer_id = 0,
                              int seq_id = 0, int token_id = 0);
    bool submit_prefetch(int request_id, int layer_id, int expert_id, double current_time_ns,
                         uint64_t bytes = DEFAULT_EXPERT_BYTES, int priority = 0, uint64_t transfer_id = 0,
                         int seq_id = 0, int token_id = 0, uint64_t candidate_id = 0);
    uint64_t submit_prefetch_id(int request_id, int layer_id, int expert_id, double current_time_ns,
                                uint64_t bytes = DEFAULT_EXPERT_BYTES, int priority = 0, uint64_t transfer_id = 0,
                                int seq_id = 0, int token_id = 0, uint64_t candidate_id = 0);

    // Dynamic priority promotion
    bool promote_transfer_to_demand(uint64_t transfer_id, double current_time_ns);

    // Explicit cancellation
    bool cancel_transfer(uint64_t transfer_id, double current_time_ns);

    // Discrete-event simulation progression
    bool step();
    void run_until(double target_time_ns);
    void run_until_empty();
    bool run_until_transfer_completed(uint64_t transfer_id);

    // Query interfaces
    double get_current_time_ns() const;
    double get_bandwidth_bytes_per_ns() const { return bandwidth_bytes_per_ns_; }
    bool is_channel_busy() const;
    bool has_active_transfer() const;
    const MemoryTransfer* get_active_transfer() const;
    const MemoryTransfer* get_transfer(uint64_t transfer_id) const;

    size_t get_demand_queue_size() const;
    size_t get_prefetch_queue_size() const;
    size_t get_paused_prefetch_queue_size() const;
    size_t get_completed_transfer_count() const;

    const TransferEngineMetrics& get_metrics() const;
    void reset();

    // Conservation invariant checker
    bool verify_conservation_invariants(double tolerance_bytes = 1e-6) const;

private:
    double bandwidth_bytes_per_ns_;
    double current_time_ns_;

    uint64_t next_transfer_id_;
    uint64_t next_event_id_;

    bool channel_busy_;
    uint64_t active_transfer_id_;

    std::deque<uint64_t> demand_queue_;
    std::deque<uint64_t> prefetch_queue_;
    std::deque<uint64_t> paused_prefetch_queue_;

    std::unordered_map<uint64_t, MemoryTransfer> transfers_;
    std::vector<uint64_t> completed_transfers_;

    // Min-heap priority queue of discrete events
    std::priority_queue<TransferEvent, std::vector<TransferEvent>, std::greater<TransferEvent>> event_queue_;

    TransferEngineMetrics metrics_;

    // Internal scheduling and state helpers
    void schedule_next_transfer();
    void start_transfer(uint64_t transfer_id);
    void preempt_active_prefetch(double arrival_timestamp_ns);
    void complete_active_transfer(double completion_timestamp_ns);
    void update_active_transfer_progress(double current_timestamp_ns);
    void schedule_completion_event(MemoryTransfer& transfer);

    bool validate_state_transition(TransferState current_state, TransferState new_state) const;
};

} // namespace llm_system

