#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::community_models::confucius4_r2t2 {

class R2T2ASRAudioEncoderGraph;
struct R2T2ASRAudioEncoderWeights;

class R2T2ASRAudioEncoderRuntime {
public:
    R2T2ASRAudioEncoderRuntime(
        std::shared_ptr<const R2T2ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~R2T2ASRAudioEncoderRuntime();

    // Actual retained graph capacity, including capacity retained after shrink.
    int64_t graph_capacity_frames() const noexcept { return graph_capacity_frames_; }

    R2T2ASRAudioEmbeddings encode(const R2T2ASRAudioFeatures & features, bool reuse_graph = false);

    // Streaming incremental encode for growing audio: embeddings of completed
    // attention windows are bitwise-frozen (block-diagonal window attention
    // plus chunk-local convolutions), so they are cached and re-validated
    // against the current mel features; only the current partial window is
    // re-encoded with an exact graph. Any condition that could differ from
    // the full reusable encode (cache cold, mel prefix changed, degenerate
    // tail) falls back to encode(features, true), keeping the output
    // bit-identical to a full re-encode. reset_streaming() drops the cache
    // when a stream restarts.
    R2T2ASRAudioEmbeddings encode_streaming(const R2T2ASRAudioFeatures & features);
    void reset_streaming();

private:
    std::shared_ptr<const R2T2ASRAssets> assets_;
    std::shared_ptr<const R2T2ASRAudioEncoderWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    int64_t graph_capacity_frames_ = 0;
    std::unique_ptr<R2T2ASRAudioEncoderGraph> graph_;
    std::vector<float> streaming_mel_;
    int64_t streaming_frames_ = 0;
    std::vector<float> streaming_embeddings_;
    int64_t streaming_tokens_ = 0;
    bool streaming_valid_ = false;
};

}  // namespace engine::community_models::confucius4_r2t2
