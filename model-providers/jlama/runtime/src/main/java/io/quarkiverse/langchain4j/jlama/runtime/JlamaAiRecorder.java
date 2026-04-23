package io.quarkiverse.langchain4j.jlama.runtime;

import java.util.function.Supplier;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.DisabledChatLanguageModel;
import dev.langchain4j.model.chat.DisabledStreamingChatLanguageModel;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.embedding.DisabledEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import io.quarkiverse.langchain4j.jlama.JlamaChatModel;
import io.quarkiverse.langchain4j.jlama.JlamaEmbeddingModel;
import io.quarkiverse.langchain4j.jlama.JlamaStreamingChatModel;
import io.quarkiverse.langchain4j.jlama.runtime.config.ChatModelConfig;
import io.quarkiverse.langchain4j.jlama.runtime.config.LangChain4jJlamaConfig;
import io.quarkiverse.langchain4j.jlama.runtime.config.LangChain4jJlamaFixedRuntimeConfig;
import io.quarkiverse.langchain4j.runtime.NamedConfigUtil;
import io.quarkus.runtime.RuntimeValue;
import io.quarkus.runtime.annotations.Recorder;

@Recorder
public class JlamaAiRecorder {

    private final RuntimeValue<LangChain4jJlamaConfig> runtimeConfig;

    public JlamaAiRecorder(RuntimeValue<LangChain4jJlamaConfig> runtimeConfig) {
        this.runtimeConfig = runtimeConfig;
    }

    public Supplier<ChatLanguageModel> chatModel(LangChain4jJlamaFixedRuntimeConfig fixedRuntimeConfig,
            String configName) {
        LangChain4jJlamaConfig config = runtimeConfig.getValue();
        LangChain4jJlamaConfig.JlamaConfig jlamaConfig = correspondingJlamaConfig(config, configName);
        LangChain4jJlamaFixedRuntimeConfig.JlamaConfig jlamaFixedRuntimeConfig = correspondingJlamaFixedRuntimeConfig(
                fixedRuntimeConfig, configName);

        if (jlamaConfig.enableIntegration()) {
            ChatModelConfig chatModelConfig = jlamaConfig.chatModel();
            String modelName = jlamaFixedRuntimeConfig.chatModel().modelName();
            var builder = JlamaChatModel.builder()
                    .modelName(modelName)
                    .modelCachePath(fixedRuntimeConfig.modelsPath());
            jlamaConfig.logRequests().ifPresent(builder::logRequests);
            jlamaConfig.logResponses().ifPresent(builder::logResponses);
            chatModelConfig.temperature().ifPresent(temp -> builder.temperature((float) temp));
            chatModelConfig.maxTokens().ifPresent(builder::maxTokens);
            return () -> builder.build();
        } else {
            return DisabledChatLanguageModel::new;
        }
    }

    public Supplier<StreamingChatLanguageModel> streamingChatModel(LangChain4jJlamaFixedRuntimeConfig fixedRuntimeConfig,
            String configName) {
        LangChain4jJlamaConfig config = runtimeConfig.getValue();
        LangChain4jJlamaConfig.JlamaConfig jlamaConfig = correspondingJlamaConfig(config, configName);
        LangChain4jJlamaFixedRuntimeConfig.JlamaConfig jlamaFixedRuntimeConfig = correspondingJlamaFixedRuntimeConfig(
                fixedRuntimeConfig, configName);

        if (jlamaConfig.enableIntegration()) {
            ChatModelConfig chatModelConfig = jlamaConfig.chatModel();
            var builder = JlamaStreamingChatModel.builder()
                    .modelName(jlamaFixedRuntimeConfig.chatModel().modelName())
                    .modelCachePath(fixedRuntimeConfig.modelsPath());
            chatModelConfig.temperature().ifPresent(temp -> builder.temperature((float) temp));
            return () -> builder.build();
        } else {
            return DisabledStreamingChatLanguageModel::new;
        }
    }

    public Supplier<EmbeddingModel> embeddingModel(LangChain4jJlamaFixedRuntimeConfig fixedRuntimeConfig,
            String configName) {
        LangChain4jJlamaConfig config = runtimeConfig.getValue();
        LangChain4jJlamaConfig.JlamaConfig jlamaConfig = correspondingJlamaConfig(config, configName);
        LangChain4jJlamaFixedRuntimeConfig.JlamaConfig jlamaFixedRuntimeConfig = correspondingJlamaFixedRuntimeConfig(
                fixedRuntimeConfig, configName);

        if (jlamaConfig.enableIntegration()) {
            var builder = JlamaEmbeddingModel.builder()
                    .modelName(jlamaFixedRuntimeConfig.embeddingModel().modelName())
                    .modelCachePath(fixedRuntimeConfig.modelsPath());
            return () -> builder.build();
        } else {
            return DisabledEmbeddingModel::new;
        }
    }

    private LangChain4jJlamaConfig.JlamaConfig correspondingJlamaConfig(LangChain4jJlamaConfig config,
            String configName) {
        return NamedConfigUtil.isDefault(configName) ? config.defaultConfig()
                : config.namedConfig().get(configName);
    }

    private LangChain4jJlamaFixedRuntimeConfig.JlamaConfig correspondingJlamaFixedRuntimeConfig(
            LangChain4jJlamaFixedRuntimeConfig runtimeConfig, String configName) {
        return NamedConfigUtil.isDefault(configName) ? runtimeConfig.defaultConfig()
                : runtimeConfig.namedConfig().get(configName);
    }
}
