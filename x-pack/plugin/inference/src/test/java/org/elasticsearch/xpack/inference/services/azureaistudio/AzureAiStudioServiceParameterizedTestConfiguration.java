/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.azureaistudio;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.ModelSecrets;
import org.elasticsearch.inference.ServiceSettings;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskSettings;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.inference.external.http.HttpClientManager;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSenderTests;
import org.elasticsearch.xpack.inference.services.AbstractInferenceServiceBaseTests;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.sparse.AlibabaCloudSearchSparseTaskSettings;
import org.elasticsearch.xpack.inference.services.azureaistudio.completion.AzureAiStudioChatCompletionServiceSettings;
import org.elasticsearch.xpack.inference.services.azureaistudio.completion.AzureAiStudioChatCompletionTaskSettings;
import org.elasticsearch.xpack.inference.services.azureaistudio.embeddings.AzureAiStudioEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.azureaistudio.embeddings.AzureAiStudioEmbeddingsTaskSettings;
import org.elasticsearch.xpack.inference.services.azureaistudio.rerank.AzureAiStudioRerankServiceSettings;
import org.elasticsearch.xpack.inference.services.azureaistudio.rerank.AzureAiStudioRerankTaskSettings;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettingsTests;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.test.ESTestCase.assertThat;
import static org.elasticsearch.test.ESTestCase.randomFrom;
import static org.elasticsearch.xpack.inference.Utils.mockClusterServiceEmpty;
import static org.elasticsearch.xpack.inference.services.ServiceComponentsTests.createWithEmptySettings;
import static org.elasticsearch.xpack.inference.services.ServiceFields.DIMENSIONS;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MAX_INPUT_TOKENS;
import static org.elasticsearch.xpack.inference.services.ServiceFields.SIMILARITY;
import static org.elasticsearch.xpack.inference.services.azureaistudio.AzureAiStudioConstants.ENDPOINT_TYPE_FIELD;
import static org.elasticsearch.xpack.inference.services.azureaistudio.AzureAiStudioConstants.PROVIDER_FIELD;
import static org.elasticsearch.xpack.inference.services.azureaistudio.AzureAiStudioConstants.TARGET_FIELD;
import static org.elasticsearch.xpack.inference.services.azureaistudio.AzureAiStudioEndpointType.REALTIME;
import static org.elasticsearch.xpack.inference.services.azureaistudio.AzureAiStudioEndpointType.TOKEN;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.Assert.fail;

public class AzureAiStudioServiceParameterizedTestConfiguration {

    public static final int REQUESTS_PER_MINUTE_VALUE = 123;
    public static final String SIMILARITY_VALUE = SimilarityMeasure.COSINE.toString();
    public static final int DIMENSIONS_VALUE = 456;
    public static final int MAX_INPUT_TOKENS_VALUE = 789;
    public static final boolean RETURN_TOKEN_VALUE = true;
    public static final String API_KEY_VALUE = "secret";
    public static final String TARGET_VALUE = "targetValue";

    public static AbstractInferenceServiceBaseTests.TestConfiguration createTestConfiguration(AzureAiStudioProvider provider) {
        var supportedTasks = switch (provider) {
            case MISTRAL, META, MICROSOFT_PHI, DATABRICKS -> EnumSet.of(TaskType.COMPLETION);
            case OPENAI -> EnumSet.of(TaskType.TEXT_EMBEDDING, TaskType.COMPLETION);
            case COHERE -> EnumSet.of(TaskType.TEXT_EMBEDDING, TaskType.COMPLETION, TaskType.RERANK);
        };
        var endpointType = switch (provider) {
            case OPENAI, COHERE -> TOKEN;
            case MISTRAL, MICROSOFT_PHI, DATABRICKS -> REALTIME;
            // Meta supports both TOKEN and REALTIME, so pick either
            case META -> randomFrom(TOKEN, REALTIME);
        };
        return new AbstractInferenceServiceBaseTests.TestConfiguration.Builder(
            new AbstractInferenceServiceBaseTests.CommonConfig(randomFrom(supportedTasks), supportedTasks, AzureAiStudioService.NAME) {

                @Override
                protected AzureAiStudioService createService(ThreadPool threadPool, HttpClientManager clientManager) {
                    var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);
                    return new AzureAiStudioService(senderFactory, createWithEmptySettings(threadPool), mockClusterServiceEmpty());
                }

                @Override
                protected Map<String, Object> createMinimalServiceSettingsMap(TaskType taskType) {
                    return new HashMap<String, Object>(
                        Map.of(
                            TARGET_FIELD,
                            TARGET_VALUE,
                            ENDPOINT_TYPE_FIELD,
                            endpointType.toString(),
                            PROVIDER_FIELD,
                            provider.toString()
                        )
                    );
                }

                @Override
                protected Map<String, Object> createAllServiceSettingsMap(TaskType taskType, ConfigurationParseContext parseContext) {
                    var minimalSettings = createMinimalServiceSettingsMap(taskType);
                    RateLimitSettingsTests.addRateLimitSettingsToMap(minimalSettings, REQUESTS_PER_MINUTE_VALUE);
                    if (taskType.equals(TaskType.TEXT_EMBEDDING)) {
                        minimalSettings.put(SIMILARITY, SIMILARITY_VALUE);
                        minimalSettings.put(DIMENSIONS, DIMENSIONS_VALUE);
                        minimalSettings.put(MAX_INPUT_TOKENS, MAX_INPUT_TOKENS_VALUE);
                    }
                    return minimalSettings;
                }

                @Override
                protected ModelSecrets createModelSecrets() {
                    return new ModelSecrets(DefaultSecretSettings.fromMap(createSecretSettingsMap()));
                }

                @Override
                protected ServiceSettings getServiceSettings(
                    Map<String, Object> serviceSettings,
                    TaskType taskType,
                    ConfigurationParseContext context
                ) {
                    return switch (taskType) {
                        case TEXT_EMBEDDING -> AzureAiStudioEmbeddingsServiceSettings.fromMap(serviceSettings, context);
                        case RERANK -> AzureAiStudioRerankServiceSettings.fromMap(serviceSettings, context);
                        case COMPLETION -> AzureAiStudioChatCompletionServiceSettings.fromMap(serviceSettings, context);
                        default -> throw new IllegalArgumentException("Unsupported task type: " + taskType);
                    };
                }

                @Override
                protected TaskSettings getEmptyTaskSettings(TaskType taskType) {
                    return switch (taskType) {
                        case TEXT_EMBEDDING -> AzureAiStudioEmbeddingsTaskSettings.fromMap(null);
                        case RERANK -> AzureAiStudioRerankTaskSettings.fromMap(null);
                        case COMPLETION -> AzureAiStudioChatCompletionTaskSettings.fromMap(null);
                        default -> throw new IllegalArgumentException("Unsupported task type: " + taskType);
                    };
                }

                @Override
                protected Map<String, Object> createAllTaskSettingsMap(TaskType taskType) {
                    var taskSettingsMap = new HashMap<String, Object>();
                    if (taskType.equals(TaskType.SPARSE_EMBEDDING) || taskType.equals(TaskType.TEXT_EMBEDDING)) {
                        if (taskType.equals(TaskType.SPARSE_EMBEDDING)) {
                            taskSettingsMap.put(AlibabaCloudSearchSparseTaskSettings.RETURN_TOKEN, RETURN_TOKEN_VALUE);
                        }
                    }
                    return taskSettingsMap;
                }

                @Override
                protected Map<String, Object> createSecretSettingsMap() {
                    return new HashMap<>(Map.of(DefaultSecretSettings.API_KEY, API_KEY_VALUE));
                }

                @Override
                protected void assertModel(Model model, TaskType taskType, boolean modelIncludesSecrets, boolean minimalSettings) {
                    assertThat(model, instanceOf(AzureAiStudioModel.class));
                    var azureAiStudioModel = (AzureAiStudioModel) model;
                    switch (taskType) {
                        case TEXT_EMBEDDING -> assertTextEmbeddingModel(azureAiStudioModel, modelIncludesSecrets, minimalSettings);
                        case RERANK -> assertRerankModel(azureAiStudioModel, modelIncludesSecrets, minimalSettings);
                        case COMPLETION -> assertCompletionModel(azureAiStudioModel, modelIncludesSecrets, minimalSettings);
                        default -> fail("unexpected task type [" + taskType + "]");
                    }
                }

                @Override
                protected EnumSet<TaskType> supportedStreamingTasks() {
                    return EnumSet.of(TaskType.CHAT_COMPLETION, TaskType.COMPLETION);
                }
            }
        ).build();
    }

    private static void assertCommonModelFields(AzureAiStudioModel azureModel, boolean modelIncludesSecrets, boolean minimalSettings) {

        if (minimalSettings) {
            assertThat(azureModel.rateLimitSettings(), is(new RateLimitSettings(240)));
        } else {
            assertThat(azureModel.rateLimitSettings(), is(new RateLimitSettings(REQUESTS_PER_MINUTE_VALUE)));
        }

        if (modelIncludesSecrets) {
            assertThat(azureModel.getSecretSettings().apiKey(), is(new SecureString(API_KEY_VALUE.toCharArray())));
        } else {
            assertThat(azureModel.getSecretSettings(), nullValue());
        }
    }

    private static void assertTextEmbeddingModel(AzureAiStudioModel model, boolean modelIncludesSecrets, boolean minimalSettings) {
        assertThat(model.getTaskType(), is(TaskType.TEXT_EMBEDDING));

        var serviceSettings = (AzureAiStudioEmbeddingsServiceSettings) model.getServiceSettings();
        assertCommonModelFields(model, modelIncludesSecrets, minimalSettings);

        var taskSettings = (AzureAiStudioEmbeddingsTaskSettings) model.getTaskSettings();
        if (minimalSettings) {
            // Check default values
            assertThat(serviceSettings.similarity(), nullValue());
            assertThat(serviceSettings.dimensions(), nullValue());
        } else {
            // Check configured values
            assertThat(serviceSettings.similarity(), is(SimilarityMeasure.fromString(SIMILARITY_VALUE)));
            assertThat(serviceSettings.dimensions(), is(DIMENSIONS_VALUE));
        }
    }

    private static void assertRerankModel(AzureAiStudioModel model, boolean modelIncludesSecrets, boolean minimalSettings) {
        assertThat(model.getTaskType(), is(TaskType.RERANK));

        var serviceSettings = (AzureAiStudioRerankServiceSettings) model.getServiceSettings();
        assertCommonModelFields(model, modelIncludesSecrets, minimalSettings);
    }

    private static void assertCompletionModel(AzureAiStudioModel model, boolean modelIncludesSecrets, boolean minimalSettings) {
        assertThat(model.getTaskType(), is(TaskType.COMPLETION));

        var serviceSettings = (AzureAiStudioChatCompletionServiceSettings) model.getServiceSettings();
        assertCommonModelFields(model, modelIncludesSecrets, minimalSettings);
    }
}
