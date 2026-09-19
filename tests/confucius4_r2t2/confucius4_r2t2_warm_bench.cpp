#include "../core/audio_task_warm_bench.h"
#include <algorithm>
#include <fstream>

namespace {
using Clock = std::chrono::steady_clock;
using J = engine::io::json::Value;
using namespace engine::tools;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
}

int main(int argc, char ** argv) {
    try {
        const auto model_path = arg_value(argc, argv, "--model", "models/Confucius4-R2T2");
        const auto audio_path = arg_value(argc, argv, "--audio", "assets/resources/sample_16k.wav");
        const auto backend = arg_value(argc, argv, "--backend", "cpu");
        const auto mode = arg_value(argc, argv, "--mode", "streaming");
        const auto language = arg_value(argc, argv, "--language", "Auto");
        const auto output = arg_value(argc, argv, "--output", "");
        const auto timing = arg_value(argc, argv, "--timing-file", "");
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int device = int_arg(argc, argv, "--device", 0);
        const int warmup = int_arg(argc, argv, "--warmup", 1);
        const int iterations = int_arg(argc, argv, "--iterations", 3);
        const int chunk_ms = int_arg(argc, argv, "--chunk-ms", 320);
        const int max_tokens = int_arg(argc, argv, "--max-tokens", 32);
        // Reject typos rather than silently benchmarking a different configuration.
        const std::vector<std::string> keys = {"--model", "--audio", "--backend", "--mode", "--language",
            "--output", "--timing-file", "--threads", "--device", "--warmup", "--iterations",
            "--chunk-ms", "--max-tokens", "--session-option"};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc || std::find(keys.begin(), keys.end(), argv[i]) == keys.end())
                throw std::runtime_error("unknown argument or missing value: " + std::string(argv[i]));
        }
        if (iterations < 1 || warmup < 0 || threads < 1 || device < 0 || max_tokens < 1 ||
            chunk_ms < 80 || chunk_ms > 2000 || (mode != "streaming" && mode != "offline") || output.empty())
            throw std::runtime_error("invalid benchmark options; --output is required");
        // The framework reads nothing from ENGINE_TIMING_*; enable the timing
        // logger directly so --timing-file actually captures stage timings.
        if (!timing.empty()) {
            const std::filesystem::path timing_path(timing);
            if (!timing_path.parent_path().empty()) std::filesystem::create_directories(timing_path.parent_path());
            engine::debug::reset_logging();
            std::ofstream clear(timing, std::ios::trunc);
            clear.close();
            if (!clear) throw std::runtime_error("cannot write timing log: " + timing);
            engine::debug::configure_logging({true, timing});
        }
        const auto audio = read_audio_buffer(audio_path);
        if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty() ||
            audio.samples.size() % audio.channels != 0) throw std::runtime_error("invalid audio");
        const double audio_ms = 1000.0 * audio.samples.size() / audio.channels / audio.sample_rate;
        engine::runtime::SessionOptions options;
        options.backend.type = backend == "metal" ? engine::core::BackendType::Metal : parse_backend(backend);
        options.backend.threads = threads;
        options.backend.device = device;
        J::Object extra;
        for (const auto & [key, value] : session_option_args(argc, argv)) {
            if (key == "confucius4_r2t2.chunk_size_ms" || key == "confucius4_r2t2.max_tokens")
                throw std::runtime_error("use --chunk-ms/--max-tokens instead of overriding them");
            options.options[key] = value;
            extra[key] = string(value);
        }
        options.options["confucius4_r2t2.chunk_size_ms"] = std::to_string(chunk_ms);
        options.options["confucius4_r2t2.max_tokens"] = std::to_string(max_tokens);
        const auto setup = Clock::now();
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load;
        load.model_path = model_path;
        load.family_hint = "confucius4_r2t2";
        auto model = registry.load(load);
        auto base = model->create_task_session({engine::runtime::VoiceTaskKind::Asr,
            mode == "streaming" ? engine::runtime::RunMode::Streaming : engine::runtime::RunMode::Offline}, options);
        auto * stream = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(base.get());
        auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(base.get());
        if ((mode == "streaming" && !stream) || (mode == "offline" && !offline))
            throw std::runtime_error("unexpected session interface");
        engine::runtime::TaskRequest request;
        request.audio_input = audio;
        request.options["language"] = language;
        request.options["max_tokens"] = std::to_string(max_tokens);
        if (mode == "streaming") stream->prepare(engine::runtime::build_preparation_request(audio));
        else offline->prepare(engine::runtime::build_preparation_request(audio));
        const double setup_ms = elapsed(setup);
        J::Array runs;
        for (int iteration = -warmup; iteration < iterations; ++iteration) {
            J::Array chunks;
            std::string delta, committed;
            double first_delta = -1, finish_ms = 0;
            Clock::time_point started;
            if (mode == "streaming") stream->set_stream_event_sink([&](const engine::runtime::StreamEvent & event) {
                if (event.partial_text && !event.partial_text->text.empty()) {
                    if (first_delta < 0) first_delta = elapsed(started);
                    delta += event.partial_text->text;
                    committed += event.partial_text->text;
                }
            });
            started = Clock::now();
            engine::runtime::TaskResult result;
            if (mode == "offline") result = offline->run(request);
            else {
                stream->start_stream(request);
                const int64_t frames = audio.samples.size() / audio.channels;
                const int64_t step = std::max<int64_t>(1, int64_t(audio.sample_rate) * chunk_ms / 1000);
                for (int64_t start = 0; start < frames; start += step) {
                    const auto take = std::min(step, frames - start);
                    engine::runtime::AudioChunk chunk;
                    chunk.sample_rate = audio.sample_rate;
                    chunk.channels = audio.channels;
                    chunk.start_sample = start;
                    const auto begin = audio.samples.begin() + start * audio.channels;
                    chunk.samples.assign(begin, begin + take * audio.channels);
                    delta.clear();
                    const auto t = Clock::now();
                    stream->process_audio_chunk(chunk);
                    const double ms = elapsed(t);
                    chunks.push_back(J::make_object({{"audio_end_ms", number(1000.0 * (start + take) / audio.sample_rate)},
                        {"wall_ms", number(ms)}, {"delta", string(delta)}}));
                }
                delta.clear();
                const auto t = Clock::now();
                result = stream->finish_stream();
                finish_ms = elapsed(t);
            }
            const double wall_ms = elapsed(started);
            if (mode == "streaming") stream->set_stream_event_sink({});
            if (!result.text_output) throw std::runtime_error("missing transcript");
            const auto & text = result.text_output->text;
            if (text.compare(0, committed.size(), committed) != 0)
                throw std::runtime_error("committed deltas are not a prefix of final text");
            if (iteration >= 0) runs.push_back(J::make_object({
                {"wall_ms", number(wall_ms)}, {"rtf", number(wall_ms / audio_ms)},
                {"first_delta_compute_ms", number(first_delta)}, {"finish_ms", number(finish_ms)},
                {"finish_delta", string(delta)}, {"text", string(text)}, {"language", string(result.text_output->language)},
                {"chunks", J::make_array(std::move(chunks))}}));
            std::cerr << (iteration < 0 ? "warmup" : "measured") << " " << iteration << ": " << wall_ms << " ms\n";
        }
        const auto report = J::make_object({{"schema_version", number(1)}, {"setup_ms", number(setup_ms)},
            {"config", J::make_object({{"model", string(std::filesystem::absolute(model_path).string())},
                {"audio", string(std::filesystem::absolute(audio_path).string())}, {"audio_ms", number(audio_ms)},
                {"backend", string(backend)}, {"mode", string(mode)}, {"language", string(language)},
                {"threads", number(threads)}, {"device", number(device)}, {"warmup", number(warmup)},
                {"iterations", number(iterations)}, {"chunk_ms", number(chunk_ms)}, {"max_tokens", number(max_tokens)},
                {"timing_enabled", number(!timing.empty())}, {"session_options", J::make_object(std::move(extra))}})},
            {"runs", J::make_array(std::move(runs))}});
        std::ofstream file(output);
        file << engine::io::json::stringify(report) << '\n';
        file.close();
        if (!file) throw std::runtime_error("cannot write report: " + output);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "r2t2 benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
