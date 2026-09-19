// P1 dependency-boundary probe for the r2t2_acc exploration.
//
// Feeds growing audio prefixes exactly like streaming does (fixed step_ms),
// then measures bitwise stability of already-seen data:
//   1. mel frames produced by the Whisper frontend (global log-mel floor may
//      move as audio grows),
//   2. audio-encoder embeddings for tokens the previous, shorter prefix
//      already produced (windowed attention should freeze completed windows).
// Reports, per step, how much of the prefix is reusable bitwise. This is
// measurement only; it asserts no transcript equality and changes no runtime
// code.
#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/audio_encoder.h"
#include "engine/community_models/confucius4_r2t2/frontend_whisper.h"
#include "engine/community_models/confucius4_r2t2/thinker.h"
#include "engine/community_models/confucius4_r2t2/tokenizer_text.h"
#include "engine/community_models/confucius4_r2t2/types.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace model = engine::community_models::confucius4_r2t2;

namespace {

struct MelDiff {
    int64_t changed_frames = 0;
    float max_abs_diff = 0.0F;
};

// R2T2ASRAudioFeatures.values holds mel as [mel_bin][frame]; frame f lives at
// values[b * frames + f] with the CURRENT frame count as stride.
MelDiff compare_mel(const model::R2T2ASRAudioFeatures & previous, const model::R2T2ASRAudioFeatures & current) {
    MelDiff diff;
    const int64_t bins = previous.mel_bins;
    for (int64_t bin = 0; bin < bins; ++bin) {
        const float * old_row = previous.values.data() + bin * previous.frames;
        const float * new_row = current.values.data() + bin * current.frames;
        for (int64_t frame = 0; frame < previous.frames; ++frame) {
            const float delta = std::abs(new_row[frame] - old_row[frame]);
            if (delta != 0.0F) {
                ++diff.changed_frames;
                diff.max_abs_diff = std::max(diff.max_abs_diff, delta);
            }
        }
    }
    return diff;
}

struct EmbeddingDiff {
    int64_t first_diff_token = -1;  // -1: common tokens are bitwise identical
    float max_abs_diff = 0.0F;
};

EmbeddingDiff compare_embeddings(const model::R2T2ASRAudioEmbeddings & previous, const model::R2T2ASRAudioEmbeddings & current) {
    EmbeddingDiff diff;
    const int64_t common = std::min(previous.tokens, current.tokens);
    const int64_t hidden = previous.hidden_size;
    for (int64_t token = 0; token < common; ++token) {
        const float * old_row = previous.values.data() + token * hidden;
        const float * new_row = current.values.data() + token * hidden;
        bool differs = false;
        for (int64_t i = 0; i < hidden; ++i) {
            const float delta = std::abs(new_row[i] - old_row[i]);
            if (delta != 0.0F) {
                differs = true;
                diff.max_abs_diff = std::max(diff.max_abs_diff, delta);
            }
        }
        if (differs && diff.first_diff_token < 0) {
            diff.first_diff_token = token;
        }
    }
    return diff;
}

}  // namespace

int main(int argc, char ** argv) {
    std::filesystem::path model_path = std::filesystem::path(ENGINE_REPO_ROOT) / "models/Confucius4-R2T2";
    std::filesystem::path audio_path = std::filesystem::path(ENGINE_REPO_ROOT) / "assets/resources/sample_16k.wav";
    engine::core::BackendConfig backend;
    backend.type = engine::core::BackendType::Metal;
    backend.threads = 8;
    int step_ms = 320;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--model") { model_path = value; }
        else if (key == "--audio") { audio_path = value; }
        else if (key == "--backend" && value == "metal") { backend.type = engine::core::BackendType::Metal; }
        else if (key == "--backend" && value == "cpu") { backend.type = engine::core::BackendType::Cpu; }
        else if (key == "--step-ms") { step_ms = std::max(80, std::stoi(value)); }
        else { std::cerr << "unknown argument: " << key << "\n"; return 1; }
    }
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "SKIP: prefix stability probe requires a Confucius4-R2T2 checkpoint\n";
        return 125;
    }
    try {
        auto assets = model::load_confucius4_r2t2_assets(model_path);
        engine::core::ExecutionContext execution(backend);
        model::R2T2ASRWhisperFrontend frontend(assets);
        model::R2T2ASRAudioEncoderRuntime encoder(assets, execution, 128ull << 20, engine::assets::TensorStorageType::Native);
        model::R2T2ASRAudioEncoderRuntime streaming_encoder(assets, execution, 128ull << 20, engine::assets::TensorStorageType::Native);
        const auto wav = engine::audio::read_wav_f32(audio_path);
        const int64_t frames_total = static_cast<int64_t>(wav.samples.size() / wav.channels);
        const int64_t step_samples = static_cast<int64_t>(static_cast<double>(wav.sample_rate) * step_ms / 1000.0);
        const auto & audio_config = assets->config.audio_encoder;
        std::cout << "step_ms=" << step_ms << " backend=" << (backend.type == engine::core::BackendType::Metal ? "metal" : "cpu")
                  << " conv_chunk_frames=" << audio_config.n_window * 2 << " n_window_infer=" << audio_config.n_window_infer << "\n";
        model::R2T2ASRAudioFeatures previous_features;
        model::R2T2ASRAudioEmbeddings previous_embeddings;
        int64_t mel_stable_steps = 0, embedding_stable_steps = 0, steps = 0;
        int64_t common_max_stable = 0;
        int64_t encoder_stream_stable = 0;
        for (int64_t start = 0; start < frames_total; start += step_samples) {
            const auto take = std::min(step_samples, frames_total - start);
            engine::runtime::AudioBuffer audio;
            audio.sample_rate = wav.sample_rate;
            audio.channels = wav.channels;
            audio.samples.assign(wav.samples.begin(), wav.samples.begin() + (start + take) * wav.channels);
            auto features = frontend.extract(audio);
            auto embeddings = encoder.encode(features, /*reuse_graph=*/true);
            auto stream_embeddings = streaming_encoder.encode_streaming(features);
            ++steps;
            bool stream_bitwise = stream_embeddings.tokens == embeddings.tokens &&
                stream_embeddings.values.size() == embeddings.values.size();
            float stream_max_diff = 0.0F;
            if (stream_bitwise) {
                for (size_t i = 0; i < embeddings.values.size(); ++i) {
                    const float delta = std::abs(stream_embeddings.values[i] - embeddings.values[i]);
                    if (delta != 0.0F) { stream_max_diff = std::max(stream_max_diff, delta); }
                }
                stream_bitwise = stream_max_diff == 0.0F;
            } else {
                stream_max_diff = -1.0F;
            }
            encoder_stream_stable += stream_bitwise ? 1 : 0;
            if (!stream_bitwise) {
                std::cout << "stream_encoder_diff audio_ms=" << (start + take) * 1000 / wav.sample_rate
                          << " tokens=" << embeddings.tokens << "/" << stream_embeddings.tokens
                          << " max_diff=" << stream_max_diff << "\n";
            }
            if (previous_embeddings.tokens > 0) {
                const auto mel = compare_mel(previous_features, features);
                const auto emb = compare_embeddings(previous_embeddings, embeddings);
                const bool mel_ok = mel.changed_frames == 0;
                const bool emb_ok = emb.first_diff_token < 0;
                mel_stable_steps += mel_ok ? 1 : 0;
                embedding_stable_steps += emb_ok ? 1 : 0;
                if (emb.first_diff_token < 0) {
                    common_max_stable = std::max(common_max_stable, previous_embeddings.tokens);
                } else {
                    common_max_stable = std::max(common_max_stable, emb.first_diff_token);
                }
                std::cout << "audio_ms=" << (start + take) * 1000 / wav.sample_rate
                          << " frames=" << features.frames << " tokens=" << embeddings.tokens
                          << " mel_changed=" << mel.changed_frames << "/" << (previous_features.frames * previous_features.mel_bins)
                          << " mel_max_diff=" << mel.max_abs_diff
                          << " emb_first_diff_token=" << emb.first_diff_token
                          << " emb_max_diff=" << emb.max_abs_diff
                          << (emb_ok ? "  emb=STABLE" : "") << (mel_ok ? " mel=STABLE" : "") << "\n";
            }
            previous_features = std::move(features);
            previous_embeddings = std::move(embeddings);
        }
        std::cout << "summary steps=" << steps
                  << " mel_prefix_bitwise_stable=" << mel_stable_steps
                  << " embedding_prefix_bitwise_stable=" << embedding_stable_steps
                  << " longest_bitwise_stable_prefix_tokens=" << common_max_stable
                  << " encoder_stream_bitwise=" << encoder_stream_stable << "/" << steps << "\n";

        // Decoder streaming-reuse parity: grow real-audio prompts the way the
        // streaming session does and require generate_streaming token output
        // to match the full-recompute generate() on every call.
        {
            model::R2T2ASRTextTokenizer tokenizer(assets);
            model::R2T2ASRThinkerRuntime reference_thinker(
                assets, execution, 256ull << 20, 256ull << 20, 64ull << 20, engine::assets::TensorStorageType::Native);
            model::R2T2ASRThinkerRuntime streaming_thinker(
                assets, execution, 256ull << 20, 256ull << 20, 64ull << 20, engine::assets::TensorStorageType::Native);
            const int64_t hidden = assets->config.text_decoder.hidden_size;
            for (const double seconds : {1.28, 2.56, 3.84, 5.12, 6.4, 7.68, 8.96, 10.24, 11.52, 12.8, 14.05}) {
                engine::runtime::AudioBuffer audio;
                audio.sample_rate = wav.sample_rate;
                audio.channels = wav.channels;
                const size_t count = std::min(wav.samples.size(), static_cast<size_t>(seconds * wav.sample_rate) * wav.channels);
                audio.samples.assign(wav.samples.begin(), wav.samples.begin() + count);
                const auto features = frontend.extract(audio);
                const auto emb_real = encoder.encode(features);
                auto prompt = tokenizer.build_prompt("", "English", emb_real.tokens);
                model::R2T2ASRGenerationOptions options;
                options.max_new_tokens = 12;
                options.reuse_graphs = true;
                const auto expected = reference_thinker.generate(prompt, emb_real, options);
                const auto actual = streaming_thinker.generate_streaming(prompt, emb_real, options);
                if (expected.token_ids.empty()) {
                    throw std::runtime_error("streaming parity fixture produced an empty reference");
                }
                if (expected.token_ids != actual.token_ids) {
                    std::cout << "stream_reuse_mismatch audio_tokens=" << emb_real.tokens
                              << " expected_tokens=" << expected.token_ids.size()
                              << " actual_tokens=" << actual.token_ids.size() << "\n";
                    for (size_t i = 0; i < std::max(expected.token_ids.size(), actual.token_ids.size()); ++i) {
                        const int32_t e = i < expected.token_ids.size() ? expected.token_ids[i] : -1;
                        const int32_t a = i < actual.token_ids.size() ? actual.token_ids[i] : -1;
                        if (e != a) { std::cout << "  first_diff_step=" << i << " expected=" << e << " actual=" << a << "\n"; break; }
                    }
                    throw std::runtime_error("streaming decoder reuse changed tokens");
                }
                std::cout << "stream_reuse_parity audio_ms=" << static_cast<int64_t>(seconds * 1000)
                          << " audio_tokens=" << emb_real.tokens
                          << " generated=" << expected.token_ids.size()
                          << " text='" << tokenizer.decode(expected.token_ids) << "'\n";
                (void)hidden;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
