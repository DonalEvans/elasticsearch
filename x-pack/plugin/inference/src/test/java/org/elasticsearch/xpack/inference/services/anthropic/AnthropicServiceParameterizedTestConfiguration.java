/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.anthropic;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.ModelSecrets;
import org.elasticsearch.inference.ServiceSettings;
import org.elasticsearch.inference.TaskSettings;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.inference.external.http.HttpClientManager;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSenderTests;
import org.elasticsearch.xpack.inference.services.AbstractInferenceServiceBaseTests;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.ServiceFields;
import org.elasticsearch.xpack.inference.services.anthropic.completion.AnthropicChatCompletionModel;
import org.elasticsearch.xpack.inference.services.anthropic.completion.AnthropicChatCompletionServiceSettings;
import org.elasticsearch.xpack.inference.services.anthropic.completion.AnthropicChatCompletionTaskSettings;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettingsTests;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.test.ESTestCase.assertThat;
import static org.elasticsearch.xpack.inference.Utils.mockClusterServiceEmpty;
import static org.elasticsearch.xpack.inference.services.ServiceComponentsTests.createWithEmptySettings;
import static org.elasticsearch.xpack.inference.services.anthropic.AnthropicServiceFields.MAX_TOKENS;
import static org.elasticsearch.xpack.inference.services.anthropic.AnthropicServiceFields.TEMPERATURE_FIELD;
import static org.elasticsearch.xpack.inference.services.anthropic.AnthropicServiceFields.TOP_K_FIELD;
import static org.elasticsearch.xpack.inference.services.anthropic.AnthropicServiceFields.TOP_P_FIELD;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.Assert.fail;

public class AnthropicServiceParameterizedTestConfiguration {

    public static final String MODEL_ID_VALUE = "model_id";
    public static final int REQUESTS_PER_MINUTE_VALUE = 123;
    public static final String API_KEY_VALUE = "secret";
    public static final int MAX_TOKENS_VALUE = 246;
    public static final double TEMPERATURE_VALUE = 0.9;
    public static final double TOP_P_VALUE = 0.8;
    public static final int TOP_K_VALUE = 456;

    public static AbstractInferenceServiceBaseTests.TestConfiguration createTestConfiguration() {
        return new AbstractInferenceServiceBaseTests.TestConfiguration.Builder(
            new AbstractInferenceServiceBaseTests.CommonConfig(
                TaskType.COMPLETION,
                EnumSet.of(TaskType.COMPLETION),
                AnthropicService.NAME
            ) {

                @Override
                protected AnthropicService createService(ThreadPool threadPool, HttpClientManager clientManager) {
                    var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);
                    return new AnthropicService(senderFactory, createWithEmptySettings(threadPool), mockClusterServiceEmpty());
                }

                @Override
                protected Map<String, Object> createMinimalServiceSettingsMap(TaskType taskType) {
                    return new HashMap<>(Map.of(ServiceFields.MODEL_ID, MODEL_ID_VALUE));
                }

                @Override
                protected Map<String, Object> createAllServiceSettingsMap(TaskType taskType, ConfigurationParseContext parseContext) {
                    var minimalSettings = createMinimalServiceSettingsMap(taskType);
                    RateLimitSettingsTests.addRateLimitSettingsToMap(minimalSettings, REQUESTS_PER_MINUTE_VALUE);
                    return minimalSettings;
                }

                @Override
                protected ModelSecrets createModelSecrets() {
                    return new ModelSecrets(DefaultSecretSettings.fromMap(createSecretSettingsMap()));
                }

                @Override
                protected Map<String, Object> createMinimalTaskSettingsMap(TaskType taskType) {
                    if (taskType == TaskType.COMPLETION) {
                        return new HashMap<>(Map.of(MAX_TOKENS, MAX_TOKENS_VALUE));
                    } else {
                        return new HashMap<>();
                    }
                }

                @Override
                protected Map<String, Object> createAllTaskSettingsMap(TaskType taskType) {
                    var taskSettings = createMinimalTaskSettingsMap(taskType);
                    if (taskType == TaskType.COMPLETION) {
                        taskSettings.put(TEMPERATURE_FIELD, TEMPERATURE_VALUE);
                        taskSettings.put(TOP_P_FIELD, TOP_P_VALUE);
                        taskSettings.put(TOP_K_FIELD, TOP_K_VALUE);
                    }
                    return taskSettings;
                }

                @Override
                protected ServiceSettings getServiceSettings(
                    Map<String, Object> serviceSettings,
                    TaskType taskType,
                    ConfigurationParseContext context
                ) {
                    if (taskType == TaskType.COMPLETION) {
                        return AnthropicChatCompletionServiceSettings.fromMap(serviceSettings, context);
                    }
                    throw new IllegalArgumentException("Unsupported task type: " + taskType);
                }

                @Override
                protected TaskSettings getEmptyTaskSettings(TaskType taskType) {
                    if (taskType == TaskType.COMPLETION) {
                        // AnthropicChatCompletionTaskSettings cannot ever be empty due to having a required setting, so return a minimal
                        // task settings instead
                        return AnthropicChatCompletionTaskSettings.fromMap(
                            createMinimalTaskSettingsMap(taskType),
                            ConfigurationParseContext.PERSISTENT
                        );
                    }
                    throw new IllegalArgumentException("Unsupported task type: " + taskType);
                }

                @Override
                protected Map<String, Object> createSecretSettingsMap() {
                    return new HashMap<>(Map.of(DefaultSecretSettings.API_KEY, API_KEY_VALUE));
                }

                @Override
                protected void assertModel(Model model, TaskType taskType, boolean modelIncludesSecrets, boolean minimalSettings) {
                    if (taskType == TaskType.COMPLETION) {
                        assertCompletionModel(model, modelIncludesSecrets, minimalSettings);
                    } else {
                        fail("unexpected task type [" + taskType + "]");
                    }
                }

                @Override
                protected EnumSet<TaskType> supportedStreamingTasks() {
                    return EnumSet.of(TaskType.COMPLETION);
                }
            }
        ).build();
    }

    private static void assertCommonModelFields(Model model, boolean modelIncludesSecrets, boolean minimalServiceSettings) {
        assertThat(model, instanceOf(AnthropicModel.class));

        var anthropicModel = (AnthropicModel) model;
        assertThat(anthropicModel.getServiceSettings().modelId(), is(MODEL_ID_VALUE));

        if (minimalServiceSettings) {
            // Check default values
            assertThat(anthropicModel.rateLimitServiceSettings().rateLimitSettings(), is(new RateLimitSettings(50)));
        } else {
            // Check configured values
            assertThat(anthropicModel.rateLimitServiceSettings().rateLimitSettings(), is(new RateLimitSettings(REQUESTS_PER_MINUTE_VALUE)));
        }

        if (modelIncludesSecrets) {
            assertThat(
                ((DefaultSecretSettings) anthropicModel.getSecretSettings()).apiKey(),
                is(new SecureString(API_KEY_VALUE.toCharArray()))
            );
        } else {
            assertThat(anthropicModel.getSecretSettings(), nullValue());
        }

    }

    private static void assertCompletionModel(Model model, boolean modelIncludesSecrets, boolean minimalServiceSettings) {
        assertCommonModelFields(model, modelIncludesSecrets, minimalServiceSettings);
        assertThat(model.getTaskType(), is(TaskType.COMPLETION));

        var completionModel = (AnthropicChatCompletionModel) model;
        assertThat(completionModel.getTaskSettings().maxTokens(), is(MAX_TOKENS_VALUE));
        if (minimalServiceSettings) {
            assertThat(completionModel.getTaskSettings().temperature(), is(nullValue()));
            assertThat(completionModel.getTaskSettings().topP(), is(nullValue()));
            assertThat(completionModel.getTaskSettings().topK(), is(nullValue()));
        } else {
            assertThat(completionModel.getTaskSettings().temperature(), is(TEMPERATURE_VALUE));
            assertThat(completionModel.getTaskSettings().topP(), is(TOP_P_VALUE));
            assertThat(completionModel.getTaskSettings().topK(), is(TOP_K_VALUE));
        }

    }
}
