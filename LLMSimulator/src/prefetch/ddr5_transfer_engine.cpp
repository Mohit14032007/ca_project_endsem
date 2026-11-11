#include "prefetch/ddr5_transfer_engine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace llm_system {

std::string to_string(TransferClass tc) {
    if (tc == TransferClass::DEMAND) {
        return "DEMAND";
    }
    return "PREFETCH";
}

std::string to_string(TransferState ts) {
    if (ts == TransferState::QUEUED) {
        return "QUEUED";
    } else if (ts == TransferState::ACTIVE) {
        return "ACTIVE";
    } else if (ts == TransferState::PAUSED) {
        return "PAUSED";
    } else if (ts == TransferState::COMPLETED) {
        return "COMPLETED";
    }
    return "CANCELLED";
}

std::string to_string(TransferEventType tet) {
    if (tet == TransferEventType::COMPLETION) {
        return "COMPLETION";
    } else if (tet == TransferEventType::DEMAND_ARRIVAL) {
        return "DEMAND_ARRIVAL";
    }
    return "PREFETCH_ARRIVAL";
}

DDR5TransferEngine::DDR5TransferEngine(double bandwidth_bytes_per_ns) {
    bandwidth_bytes_per_ns_ = bandwidth_bytes_per_ns;
    current_time_ns_ = 0.0;
    next_transfer_id_ = 1;
    next_event_id_ = 1;
    channel_busy_ = false;
    active_transfer_id_ = 0;
}

void DDR5TransferEngine::reset() {
    current_time_ns_ = 0.0;
    next_transfer_id_ = 1;
    next_event_id_ = 1;
    channel_busy_ = false;
    active_transfer_id_ = 0;

    demand_queue_.clear();
    prefetch_queue_.clear();
    paused_prefetch_queue_.clear();
    transfers_.clear();
    completed_transfers_.clear();

    while (!event_queue_.empty()) {
        event_queue_.pop();
    }

    metrics_ = TransferEngineMetrics();
}

bool DDR5TransferEngine::validate_state_transition(TransferState current_state, TransferState new_state) const {
    if (current_state == TransferState::QUEUED) {
        if (new_state == TransferState::ACTIVE) {
            return true;
        }
        if (new_state == TransferState::CANCELLED) {
            return true;
        }
        return false;
    }

    if (current_state == TransferState::ACTIVE) {
        if (new_state == TransferState::PAUSED) {
            return true;
        }
        if (new_state == TransferState::COMPLETED) {
            return true;
        }
        if (new_state == TransferState::CANCELLED) {
            return true;
        }
        return false;
    }

    if (current_state == TransferState::PAUSED) {
        if (new_state == TransferState::ACTIVE) {
            return true;
        }
        if (new_state == TransferState::CANCELLED) {
            return true;
        }
        return false;
    }

    // COMPLETED and CANCELLED are terminal states
    return false;
}

bool DDR5TransferEngine::submit_transfer(const MemoryTransfer& transfer_spec, double current_time_ns) {
    if (transfer_spec.total_bytes == 0) {
        std::cerr << "[ERROR][DDR5TransferEngine] Rejection: Zero-byte transfer submission attempted.\n";
        return false;
    }

    uint64_t tid = transfer_spec.transfer_id;
    if (tid == 0) {
        tid = next_transfer_id_++;
    } else {
        if (transfers_.find(tid) != transfers_.end()) {
            std::cerr << "[ERROR][DDR5TransferEngine] Rejection: Duplicate transfer ID " << tid << " submitted.\n";
            return false;
        }
        if (tid >= next_transfer_id_) {
            next_transfer_id_ = tid + 1;
        }
    }

    // Advance simulator to the exact arrival timestamp
    if (current_time_ns > current_time_ns_) {
        run_until(current_time_ns);
    }
    current_time_ns_ = current_time_ns;

    MemoryTransfer transfer = transfer_spec;
    transfer.transfer_id = tid;
    transfer.remaining_bytes = static_cast<double>(transfer.total_bytes);
    transfer.transferred_bytes = 0.0;
    transfer.arrival_time_ns = current_time_ns;
    transfer.service_start_time_ns = -1.0;
    transfer.last_start_time_ns = -1.0;
    transfer.completion_time_ns = -1.0;
    transfer.state = TransferState::QUEUED;
    transfer.generation = 1;
    transfer.completed_tensors = 0;

    transfers_[tid] = transfer;

    if (transfer.transfer_class == TransferClass::DEMAND) {
        metrics_.total_demand_submitted++;

        // If an active prefetch is currently transmitting, preempt it immediately
        if (channel_busy_ && active_transfer_id_ != 0) {
            auto it = transfers_.find(active_transfer_id_);
            if (it != transfers_.end()) {
                if (it->second.transfer_class == TransferClass::PREFETCH) {
                    preempt_active_prefetch(current_time_ns);
                }
            }
        }

        demand_queue_.push_back(tid);
    } else {
        metrics_.total_prefetch_submitted++;
        prefetch_queue_.push_back(tid);
    }

    if (!channel_busy_) {
        schedule_next_transfer();
    }

    return true;
}

bool DDR5TransferEngine::submit_demand(int request_id, int layer_id, int expert_id, double current_time_ns,
                                       uint64_t bytes, uint64_t transfer_id,
                                       int seq_id, int token_id) {
    MemoryTransfer t;
    t.transfer_id = transfer_id;
    t.request_id = request_id;
    t.seq_id = seq_id;
    t.token_id = token_id;
    t.layer_id = layer_id;
    t.expert_id = expert_id;
    t.transfer_class = TransferClass::DEMAND;
    t.total_bytes = bytes;
    t.num_tensors = DEFAULT_TENSORS_PER_EXPERT;
    t.tensor_size = DEFAULT_TENSOR_BYTES;
    return submit_transfer(t, current_time_ns);
}

uint64_t DDR5TransferEngine::submit_demand_id(int request_id, int layer_id, int expert_id, double current_time_ns,
                                              uint64_t bytes, uint64_t transfer_id,
                                              int seq_id, int token_id) {
    uint64_t tid = transfer_id;
    if (tid == 0) {
        tid = next_transfer_id_;
    }
    bool ok = submit_demand(request_id, layer_id, expert_id, current_time_ns, bytes, tid, seq_id, token_id);
    if (!ok) {
        return 0;
    }
    return tid;
}

bool DDR5TransferEngine::submit_prefetch(int request_id, int layer_id, int expert_id, double current_time_ns,
                                         uint64_t bytes, int priority, uint64_t transfer_id,
                                         int seq_id, int token_id, uint64_t candidate_id) {
    MemoryTransfer t;
    t.transfer_id = transfer_id;
    t.request_id = request_id;
    t.seq_id = seq_id;
    t.token_id = token_id;
    t.candidate_id = candidate_id;
    t.layer_id = layer_id;
    t.expert_id = expert_id;
    t.transfer_class = TransferClass::PREFETCH;
    t.total_bytes = bytes;
    t.priority = priority;
    t.num_tensors = DEFAULT_TENSORS_PER_EXPERT;
    t.tensor_size = DEFAULT_TENSOR_BYTES;
    return submit_transfer(t, current_time_ns);
}

uint64_t DDR5TransferEngine::submit_prefetch_id(int request_id, int layer_id, int expert_id, double current_time_ns,
                                                uint64_t bytes, int priority, uint64_t transfer_id,
                                                int seq_id, int token_id, uint64_t candidate_id) {
    uint64_t tid = transfer_id;
    if (tid == 0) {
        tid = next_transfer_id_;
    }
    bool ok = submit_prefetch(request_id, layer_id, expert_id, current_time_ns, bytes, priority, tid, seq_id, token_id, candidate_id);
    if (!ok) {
        return 0;
    }
    return tid;
}

bool DDR5TransferEngine::promote_transfer_to_demand(uint64_t transfer_id, double current_time_ns) {
    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end()) {
        return false;
    }

    MemoryTransfer& transfer = it->second;
    if (transfer.state == TransferState::CANCELLED) {
        return false;
    }

    if (current_time_ns > current_time_ns_) {
        run_until(current_time_ns);
    }
    current_time_ns_ = current_time_ns;

    if (transfer.state == TransferState::COMPLETED) {
        return true;
    }

    if (transfer.transfer_class == TransferClass::DEMAND) {
        return true;
    }

    transfer.transfer_class = TransferClass::DEMAND;

    if (transfer.state == TransferState::ACTIVE) {
        update_active_transfer_progress(current_time_ns);
        return true;
    }

    if (transfer.state == TransferState::QUEUED) {
        for (auto q_it = prefetch_queue_.begin(); q_it != prefetch_queue_.end(); ++q_it) {
            if (*q_it == transfer_id) {
                prefetch_queue_.erase(q_it);
                break;
            }
        }
        demand_queue_.push_back(transfer_id);

        if (channel_busy_ && active_transfer_id_ != 0) {
            auto act_it = transfers_.find(active_transfer_id_);
            if (act_it != transfers_.end()) {
                if (act_it->second.transfer_class == TransferClass::PREFETCH) {
                    preempt_active_prefetch(current_time_ns);
                    schedule_next_transfer();
                }
            }
        } else if (!channel_busy_) {
            schedule_next_transfer();
        }
        return true;
    }

    if (transfer.state == TransferState::PAUSED) {
        for (auto q_it = paused_prefetch_queue_.begin(); q_it != paused_prefetch_queue_.end(); ++q_it) {
            if (*q_it == transfer_id) {
                paused_prefetch_queue_.erase(q_it);
                break;
            }
        }
        demand_queue_.push_back(transfer_id);

        if (channel_busy_ && active_transfer_id_ != 0) {
            auto act_it = transfers_.find(active_transfer_id_);
            if (act_it != transfers_.end()) {
                if (act_it->second.transfer_class == TransferClass::PREFETCH) {
                    preempt_active_prefetch(current_time_ns);
                    schedule_next_transfer();
                }
            }
        } else if (!channel_busy_) {
            schedule_next_transfer();
        }
        return true;
    }

    return false;
}

bool DDR5TransferEngine::cancel_transfer(uint64_t transfer_id, double current_time_ns) {
    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end()) {
        return false;
    }

    MemoryTransfer& transfer = it->second;
    if (transfer.state == TransferState::COMPLETED || transfer.state == TransferState::CANCELLED) {
        return false;
    }

    if (current_time_ns > current_time_ns_) {
        run_until(current_time_ns);
    }
    current_time_ns_ = current_time_ns;

    if (transfer.state == TransferState::ACTIVE) {
        update_active_transfer_progress(current_time_ns);
        transfer.state = TransferState::CANCELLED;
        transfer.generation++;
        channel_busy_ = false;
        active_transfer_id_ = 0;
        metrics_.cancelled_transfers++;
        schedule_next_transfer();
        return true;
    }

    if (transfer.state == TransferState::QUEUED) {
        transfer.state = TransferState::CANCELLED;
        transfer.generation++;

        if (transfer.transfer_class == TransferClass::DEMAND) {
            for (auto q_it = demand_queue_.begin(); q_it != demand_queue_.end(); ++q_it) {
                if (*q_it == transfer_id) {
                    demand_queue_.erase(q_it);
                    break;
                }
            }
        } else {
            for (auto q_it = prefetch_queue_.begin(); q_it != prefetch_queue_.end(); ++q_it) {
                if (*q_it == transfer_id) {
                    prefetch_queue_.erase(q_it);
                    break;
                }
            }
        }

        metrics_.cancelled_transfers++;
        return true;
    }

    if (transfer.state == TransferState::PAUSED) {
        transfer.state = TransferState::CANCELLED;
        transfer.generation++;

        for (auto q_it = paused_prefetch_queue_.begin(); q_it != paused_prefetch_queue_.end(); ++q_it) {
            if (*q_it == transfer_id) {
                paused_prefetch_queue_.erase(q_it);
                break;
            }
        }

        metrics_.cancelled_transfers++;
        return true;
    }

    return false;
}

void DDR5TransferEngine::update_active_transfer_progress(double current_timestamp_ns) {
    if (!channel_busy_ || active_transfer_id_ == 0) {
        return;
    }

    auto it = transfers_.find(active_transfer_id_);
    if (it == transfers_.end()) {
        return;
    }

    MemoryTransfer& active = it->second;
    if (current_timestamp_ns <= active.last_start_time_ns) {
        return;
    }

    double delta_t = current_timestamp_ns - active.last_start_time_ns;
    double delta_bytes = delta_t * bandwidth_bytes_per_ns_;

    if (delta_bytes > active.remaining_bytes) {
        delta_bytes = active.remaining_bytes;
    }

    active.transferred_bytes += delta_bytes;
    active.remaining_bytes -= delta_bytes;

    if (active.remaining_bytes < 0.0) {
        active.remaining_bytes = 0.0;
    }

    if (active.tensor_size > 0) {
        int done = static_cast<int>(active.transferred_bytes / static_cast<double>(active.tensor_size));
        if (done > active.num_tensors) {
            done = active.num_tensors;
        }
        active.completed_tensors = done;
    }

    if (active.transfer_class == TransferClass::DEMAND) {
        metrics_.demand_service_time_ns += delta_t;
    } else {
        metrics_.prefetch_service_time_ns += delta_t;
    }

    active.last_start_time_ns = current_timestamp_ns;
}

void DDR5TransferEngine::preempt_active_prefetch(double arrival_timestamp_ns) {
    if (!channel_busy_ || active_transfer_id_ == 0) {
        return;
    }

    auto it = transfers_.find(active_transfer_id_);
    if (it == transfers_.end()) {
        return;
    }

    MemoryTransfer& active = it->second;
    if (active.transfer_class != TransferClass::PREFETCH) {
        return;
    }

    update_active_transfer_progress(arrival_timestamp_ns);

    // If remaining bytes reached zero exactly at this timestamp, complete it instead of pausing
    if (active.remaining_bytes <= 1e-9) {
        complete_active_transfer(arrival_timestamp_ns);
        return;
    }

    // Invalidate old completion event by incrementing generation counter
    active.state = TransferState::PAUSED;
    active.completion_time_ns = -1.0;
    active.generation++;

    // Push paused prefetch to the head of paused queue for resumption
    paused_prefetch_queue_.push_front(active.transfer_id);

    metrics_.prefetch_preemptions++;
    channel_busy_ = false;
    active_transfer_id_ = 0;
}

void DDR5TransferEngine::complete_active_transfer(double completion_timestamp_ns) {
    current_time_ns_ = completion_timestamp_ns;

    if (!channel_busy_ || active_transfer_id_ == 0) {
        return;
    }

    auto it = transfers_.find(active_transfer_id_);
    if (it == transfers_.end()) {
        channel_busy_ = false;
        active_transfer_id_ = 0;
        return;
    }

    MemoryTransfer& active = it->second;

    double delta_t = completion_timestamp_ns - active.last_start_time_ns;
    if (delta_t > 0.0) {
        if (active.transfer_class == TransferClass::DEMAND) {
            metrics_.demand_service_time_ns += delta_t;
        } else {
            metrics_.prefetch_service_time_ns += delta_t;
        }
    }

    active.transferred_bytes = static_cast<double>(active.total_bytes);
    active.remaining_bytes = 0.0;
    active.completed_tensors = active.num_tensors;
    active.completion_time_ns = completion_timestamp_ns;
    active.state = TransferState::COMPLETED;
    active.generation++;

    if (active.transfer_class == TransferClass::DEMAND) {
        metrics_.demand_completions++;
    } else {
        metrics_.prefetch_completions++;
    }

    metrics_.total_bytes_transferred += active.total_bytes;
    completed_transfers_.push_back(active.transfer_id);

    channel_busy_ = false;
    active_transfer_id_ = 0;

    // Immediately schedule next eligible transfer
    schedule_next_transfer();
}

void DDR5TransferEngine::start_transfer(uint64_t transfer_id) {
    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end()) {
        return;
    }

    MemoryTransfer& transfer = it->second;

    if (transfer.state == TransferState::PAUSED) {
        metrics_.prefetch_resumptions++;
    } else if (transfer.state == TransferState::QUEUED) {
        double wait_time = current_time_ns_ - transfer.arrival_time_ns;
        if (wait_time < 0.0) {
            wait_time = 0.0;
        }
        if (transfer.transfer_class == TransferClass::DEMAND) {
            metrics_.demand_queue_waiting_time_ns += wait_time;
        } else {
            metrics_.prefetch_queue_waiting_time_ns += wait_time;
        }
    }

    transfer.state = TransferState::ACTIVE;
    transfer.generation++;

    if (transfer.service_start_time_ns < 0.0) {
        transfer.service_start_time_ns = current_time_ns_;
    }
    transfer.last_start_time_ns = current_time_ns_;

    channel_busy_ = true;
    active_transfer_id_ = transfer.transfer_id;

    schedule_completion_event(transfer);
}

void DDR5TransferEngine::schedule_completion_event(MemoryTransfer& transfer) {
    double duration_ns = transfer.remaining_bytes / bandwidth_bytes_per_ns_;
    double completion_time = current_time_ns_ + duration_ns;
    transfer.completion_time_ns = completion_time;

    TransferEvent event;
    event.event_id = next_event_id_++;
    event.timestamp_ns = completion_time;
    event.type = TransferEventType::COMPLETION;
    event.transfer_id = transfer.transfer_id;
    event.generation = transfer.generation;

    event_queue_.push(event);
}

void DDR5TransferEngine::schedule_next_transfer() {
    if (channel_busy_) {
        return;
    }

    // Priority 1: Strict FIFO service for Demand requests
    if (!demand_queue_.empty()) {
        uint64_t tid = demand_queue_.front();
        demand_queue_.pop_front();
        start_transfer(tid);
        return;
    }

    // Priority 2: Resumption of Paused Prefetch transfers (work-preserving)
    if (!paused_prefetch_queue_.empty()) {
        uint64_t tid = paused_prefetch_queue_.front();
        paused_prefetch_queue_.pop_front();
        start_transfer(tid);
        return;
    }

    // Priority 3: Queued Prefetch candidates
    if (!prefetch_queue_.empty()) {
        uint64_t tid = prefetch_queue_.front();
        prefetch_queue_.pop_front();
        start_transfer(tid);
        return;
    }
}

bool DDR5TransferEngine::step() {
    if (event_queue_.empty()) {
        return false;
    }

    TransferEvent event = event_queue_.top();
    event_queue_.pop();

    current_time_ns_ = event.timestamp_ns;

    auto it = transfers_.find(event.transfer_id);
    if (it == transfers_.end()) {
        metrics_.stale_events_ignored++;
        return true;
    }

    MemoryTransfer& transfer = it->second;

    // Stale-event protection:
    // Event is valid ONLY if transfer is ACTIVE, matches generation, and is active on the channel
    if (transfer.state != TransferState::ACTIVE ||
        transfer.generation != event.generation ||
        transfer.transfer_id != active_transfer_id_) {
        metrics_.stale_events_ignored++;
        return true;
    }

    if (event.type == TransferEventType::COMPLETION) {
        complete_active_transfer(event.timestamp_ns);
    }

    return true;
}

void DDR5TransferEngine::run_until(double target_time_ns) {
    while (!event_queue_.empty()) {
        const TransferEvent& next_ev = event_queue_.top();
        if (next_ev.timestamp_ns > target_time_ns) {
            break;
        }
        step();
    }
    if (current_time_ns_ < target_time_ns) {
        if (channel_busy_ && active_transfer_id_ != 0) {
            update_active_transfer_progress(target_time_ns);
        }
        current_time_ns_ = target_time_ns;
    }
}

void DDR5TransferEngine::run_until_empty() {
    while (!event_queue_.empty()) {
        step();
    }
}

bool DDR5TransferEngine::run_until_transfer_completed(uint64_t transfer_id) {
    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end()) {
        return false;
    }
    while (it->second.state != TransferState::COMPLETED && it->second.state != TransferState::CANCELLED) {
        if (event_queue_.empty()) {
            return false;
        }
        step();
        it = transfers_.find(transfer_id);
        if (it == transfers_.end()) {
            return false;
        }
    }
    if (it->second.state == TransferState::COMPLETED) {
        return true;
    }
    return false;
}

double DDR5TransferEngine::get_current_time_ns() const {
    return current_time_ns_;
}

bool DDR5TransferEngine::is_channel_busy() const {
    return channel_busy_;
}

bool DDR5TransferEngine::has_active_transfer() const {
    return channel_busy_;
}

const MemoryTransfer* DDR5TransferEngine::get_active_transfer() const {
    if (!channel_busy_ || active_transfer_id_ == 0) {
        return nullptr;
    }
    auto it = transfers_.find(active_transfer_id_);
    if (it != transfers_.end()) {
        return &(it->second);
    }
    return nullptr;
}

const MemoryTransfer* DDR5TransferEngine::get_transfer(uint64_t transfer_id) const {
    auto it = transfers_.find(transfer_id);
    if (it != transfers_.end()) {
        return &(it->second);
    }
    return nullptr;
}

size_t DDR5TransferEngine::get_demand_queue_size() const {
    return demand_queue_.size();
}

size_t DDR5TransferEngine::get_prefetch_queue_size() const {
    return prefetch_queue_.size();
}

size_t DDR5TransferEngine::get_paused_prefetch_queue_size() const {
    return paused_prefetch_queue_.size();
}

size_t DDR5TransferEngine::get_completed_transfer_count() const {
    return completed_transfers_.size();
}

const TransferEngineMetrics& DDR5TransferEngine::get_metrics() const {
    return metrics_;
}

bool DDR5TransferEngine::verify_conservation_invariants(double tolerance_bytes) const {
    for (const auto& pair : transfers_) {
        const MemoryTransfer& t = pair.second;
        if (t.state == TransferState::CANCELLED) {
            continue;
        }

        // Invariant 1: total_bytes == transferred_bytes + remaining_bytes
        double sum = t.transferred_bytes + t.remaining_bytes;
        double diff = std::abs(static_cast<double>(t.total_bytes) - sum);
        if (diff > tolerance_bytes) {
            std::cerr << "[INVARIANT FAIL] Byte conservation violated for Transfer " << t.transfer_id
                      << ": total=" << t.total_bytes << ", transferred=" << t.transferred_bytes
                      << ", remaining=" << t.remaining_bytes << ", diff=" << diff << "\n";
            return false;
        }

        // Invariant 2: Non-negative remaining and transferred bytes
        if (t.remaining_bytes < 0.0) {
            std::cerr << "[INVARIANT FAIL] Negative remaining bytes for Transfer " << t.transfer_id << "\n";
            return false;
        }
        if (t.transferred_bytes < 0.0) {
            std::cerr << "[INVARIANT FAIL] Negative transferred bytes for Transfer " << t.transfer_id << "\n";
            return false;
        }

        // Invariant 3: Completed transfers must have exactly 0 remaining bytes
        if (t.state == TransferState::COMPLETED) {
            if (t.remaining_bytes > tolerance_bytes) {
                std::cerr << "[INVARIANT FAIL] Completed transfer " << t.transfer_id
                          << " has non-zero remaining bytes: " << t.remaining_bytes << "\n";
                return false;
            }
        }

        // Invariant 4: Queued transfers must have 0 transferred bytes
        if (t.state == TransferState::QUEUED) {
            if (t.transferred_bytes > tolerance_bytes) {
                std::cerr << "[INVARIANT FAIL] Queued transfer " << t.transfer_id
                          << " has non-zero transferred bytes: " << t.transferred_bytes << "\n";
                return false;
            }
        }
    }

    // Invariant 5: Channel busy consistency
    if (channel_busy_) {
        if (active_transfer_id_ == 0) {
            std::cerr << "[INVARIANT FAIL] Channel marked busy but active_transfer_id is 0\n";
            return false;
        }
        auto it = transfers_.find(active_transfer_id_);
        if (it == transfers_.end()) {
            std::cerr << "[INVARIANT FAIL] Active transfer ID " << active_transfer_id_ << " not found in transfers map\n";
            return false;
        }
        if (it->second.state != TransferState::ACTIVE) {
            std::cerr << "[INVARIANT FAIL] Active transfer ID " << active_transfer_id_ << " has non-ACTIVE state: "
                      << to_string(it->second.state) << "\n";
            return false;
        }
    } else {
        if (active_transfer_id_ != 0) {
            std::cerr << "[INVARIANT FAIL] Channel marked idle but active_transfer_id is " << active_transfer_id_ << "\n";
            return false;
        }
    }

    // Invariant 6: Demand priority invariant - no prefetch active while demand queued
    if (channel_busy_ && active_transfer_id_ != 0) {
        auto it = transfers_.find(active_transfer_id_);
        if (it != transfers_.end()) {
            if (it->second.transfer_class == TransferClass::PREFETCH) {
                if (!demand_queue_.empty()) {
                    std::cerr << "[INVARIANT FAIL] Prefetch active while demand queue is non-empty\n";
                    return false;
                }
            }
        }
    }

    return true;
}

} // namespace llm_system

