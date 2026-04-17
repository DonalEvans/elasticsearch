/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.alibabacloudsearch;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.inference.InputType;
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
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.completion.AlibabaCloudSearchCompletionServiceSettings;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.completion.AlibabaCloudSearchCompletionTaskSettings;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.embeddings.AlibabaCloudSearchEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.embeddings.AlibabaCloudSearchEmbeddingsTaskSettings;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.rerank.AlibabaCloudSearchRerankServiceSettings;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.rerank.AlibabaCloudSearchRerankTaskSettings;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.sparse.AlibabaCloudSearchSparseServiceSettings;
import org.elasticsearch.xpack.inference.services.alibabacloudsearch.sparse.AlibabaCloudSearchSparseTaskSettings;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettingsTests;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.test.ESTestCase.assertThat;
import static org.elasticsearch.xpack.inference.Utils.mockClusterServiceEmpty;
import static org.elasticsearch.xpack.inference.services.ServiceComponentsTests.createWithEmptySettings;
import static org.elasticsearch.xpack.inference.services.ServiceFields.DIMENSIONS;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MAX_INPUT_TOKENS;
import static org.elasticsearch.xpack.inference.services.ServiceFields.SIMILARITY;
import static org.elasticsearch.xpack.inference.services.alibabacloudsearch.AlibabaCloudSearchServiceSettings.HOST;
import static org.elasticsearch.xpack.inference.services.alibabacloudsearch.AlibabaCloudSearchServiceSettings.HTTP_SCHEMA_NAME;
import static org.elasticsearch.xpack.inference.services.alibabacloudsearch.AlibabaCloudSearchServiceSettings.SERVICE_ID;
import static org.elasticsearch.xpack.inference.services.alibabacloudsearch.AlibabaCloudSearchServiceSettings.WORKSPACE_NAME;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.Assert.fail;

public class AlibabaCloudSearchServiceParameterizedTestConfiguration {

    public static final String SERVICE_ID_VALUE = "serviceId";
    public static final String HOST_VALUE = "someHost";
    public static final String WORKSPACE_NAME_VALUE = "workspaceName";
    public static final int REQUESTS_PER_MINUTE_VALUE = 123;
    public static final String HTTP_SCHEMA_VALUE = "http";
    public static final String SIMILARITY_VALUE = SimilarityMeasure.COSINE.toString();
    public static final int DIMENSIONS_VALUE = 456;
    public static final int MAX_INPUT_TOKENS_VALUE = 789;
    public static final String INPUT_TYPE_VALUE = InputType.SEARCH.toString();
    public static final boolean RETURN_TOKEN_VALUE = true;
    public static final String API_KEY_VALUE = "secret";

    public static AbstractInferenceServiceBaseTests.TestConfiguration createTestConfiguration() {
        return new AbstractInferenceServiceBaseTests.TestConfiguration.Builder(
            new AbstractInferenceServiceBaseTests.CommonConfig(
                TaskType.TEXT_EMBEDDING,
                EnumSet.of(TaskType.TEXT_EMBEDDING, TaskType.SPARSE_EMBEDDING, TaskType.RERANK, TaskType.COMPLETION),
                AlibabaCloudSearchService.NAME
            ) {

                @Override
                protected AlibabaCloudSearchService createService(ThreadPool threadPool, HttpClientManager clientManager) {
                    var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);
                    return new AlibabaCloudSearchService(senderFactory, createWithEmptySettings(threadPool), mockClusterServiceEmpty());
                }

                @Override
                protected Map<String, Object> createMinimalServiceSettingsMap(TaskType taskType) {
                    return new HashMap<>(Map.of(SERVICE_ID, SERVICE_ID_VALUE, HOST, HOST_VALUE, WORKSPACE_NAME, WORKSPACE_NAME_VALUE));
                }

                @Override
                protected Map<String, Object> createAllSupportedServiceSettingsMap(
                    TaskType taskType,
                    ConfigurationParseContext parseContext
                ) {
                    var minimalSettings = createMinimalServiceSettingsMap(taskType);
                    RateLimitSettingsTests.addRateLimitSettingsToMap(minimalSettings, REQUESTS_PER_MINUTE_VALUE);
                    minimalSettings.put(HTTP_SCHEMA_NAME, HTTP_SCHEMA_VALUE);
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
                        case TEXT_EMBEDDING -> AlibabaCloudSearchEmbeddingsServiceSettings.fromMap(serviceSettings, context);
                        case SPARSE_EMBEDDING -> AlibabaCloudSearchSparseServiceSettings.fromMap(serviceSettings, context);
                        case RERANK -> AlibabaCloudSearchRerankServiceSettings.fromMap(serviceSettings, context);
                        case COMPLETION -> AlibabaCloudSearchCompletionServiceSettings.fromMap(serviceSettings, context);
                        default -> throw new IllegalArgumentException("Unsupported task type: " + taskType);
                    };
                }

                @Override
                protected TaskSettings getEmptyTaskSettings(TaskType taskType) {
                    return switch (taskType) {
                        case TEXT_EMBEDDING -> AlibabaCloudSearchEmbeddingsTaskSettings.fromMap(null);
                        case SPARSE_EMBEDDING -> AlibabaCloudSearchSparseTaskSettings.fromMap(null);
                        case RERANK -> AlibabaCloudSearchRerankTaskSettings.fromMap(null);
                        case COMPLETION -> AlibabaCloudSearchCompletionTaskSettings.fromMap(null);
                        default -> throw new IllegalArgumentException("Unsupported task type: " + taskType);
                    };
                }

                @Override
                protected Map<String, Object> createTaskSettingsMap(TaskType taskType) {
                    var taskSettingsMap = new HashMap<String, Object>();
                    if (taskType.equals(TaskType.SPARSE_EMBEDDING) || taskType.equals(TaskType.TEXT_EMBEDDING)) {
                        if (taskType.equals(TaskType.SPARSE_EMBEDDING)) {
                            taskSettingsMap.put(AlibabaCloudSearchSparseTaskSettings.RETURN_TOKEN, RETURN_TOKEN_VALUE);
                        }
                        taskSettingsMap.put(AlibabaCloudSearchEmbeddingsTaskSettings.INPUT_TYPE, INPUT_TYPE_VALUE);
                    }
                    return taskSettingsMap;
                }

                @Override
                protected Map<String, Object> createSecretSettingsMap() {
                    return new HashMap<>(Map.of(DefaultSecretSettings.API_KEY, API_KEY_VALUE));
                }

                @Override
                protected void assertModel(Model model, TaskType taskType, boolean modelIncludesSecrets, boolean minimalSettings) {
                    assertThat(model, instanceOf(AlibabaCloudSearchModel.class));
                    var alibabaModel = (AlibabaCloudSearchModel) model;
                    switch (taskType) {
                        case TEXT_EMBEDDING -> assertTextEmbeddingModel(alibabaModel, modelIncludesSecrets, minimalSettings);
                        case SPARSE_EMBEDDING -> assertSparseEmbeddingModel(alibabaModel, modelIncludesSecrets, minimalSettings);
                        case RERANK -> assertRerankModel(alibabaModel, modelIncludesSecrets, minimalSettings);
                        case COMPLETION -> assertCompletionModel(alibabaModel, modelIncludesSecrets, minimalSettings);
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

    private static void assertCommonModelFields(
        AlibabaCloudSearchModel alibabaModel,
        AlibabaCloudSearchServiceSettings commonServiceSettings,
        boolean modelIncludesSecrets,
        boolean minimalSettings
    ) {
        // Alibaba Cloud Search returns the service ID from the modelId() method
        assertThat(commonServiceSettings.modelId(), is(SERVICE_ID_VALUE));
        assertThat(commonServiceSettings.getHost(), is(HOST_VALUE));
        assertThat(commonServiceSettings.getWorkspaceName(), is(WORKSPACE_NAME_VALUE));

        if (minimalSettings) {
            assertThat(commonServiceSettings.getHttpSchema(), nullValue());
            assertThat(commonServiceSettings.rateLimitSettings(), is(new RateLimitSettings(1000)));
        } else {
            assertThat(commonServiceSettings.getHttpSchema(), is(HTTP_SCHEMA_VALUE));
            assertThat(commonServiceSettings.rateLimitSettings(), is(new RateLimitSettings(REQUESTS_PER_MINUTE_VALUE)));
        }

        if (modelIncludesSecrets) {
            assertThat(
                ((DefaultSecretSettings) alibabaModel.getSecretSettings()).apiKey(),
                is(new SecureString(API_KEY_VALUE.toCharArray()))
            );
        } else {
            assertThat(alibabaModel.getSecretSettings(), nullValue());
        }
    }

    private static void assertTextEmbeddingModel(AlibabaCloudSearchModel model, boolean modelIncludesSecrets, boolean minimalSettings) {
        assertThat(model.getTaskType(), is(TaskType.TEXT_EMBEDDING));

        var serviceSettings = (AlibabaCloudSearchEmbeddingsServiceSettings) model.getServiceSettings();
        assertCommonModelFields(model, serviceSettings.getCommonSettings(), modelIncludesSecrets, minimalSettings);

        var taskSettings = (AlibabaCloudSearchEmbeddingsTaskSettings) model.getTaskSettings();
        if (minimalSettings) {
            // Check default values
            assertThat(serviceSettings.similarity(), nullValue());
            assertThat(serviceSettings.dimensions(), nullValue());
            assertThat(serviceSettings.getMaxInputTokens(), nullValue());
            assertThat(taskSettings.getInputType(), nullValue());
        } else {
            // Check configured values
            assertThat(serviceSettings.similarity(), is(SimilarityMeasure.fromString(SIMILARITY_VALUE)));
            assertThat(serviceSettings.dimensions(), is(DIMENSIONS_VALUE));
            assertThat(serviceSettings.getMaxInputTokens(), is(MAX_INPUT_TOKENS_VALUE));
            assertThat(taskSettings.getInputType(), is(InputType.fromString(INPUT_TYPE_VALUE)));
        }
    }

    private static void assertSparseEmbeddingModel(AlibabaCloudSearchModel model, boolean modelIncludesSecrets, boolean minimalSettings) {
        assertThat(model.getTaskType(), is(TaskType.SPARSE_EMBEDDING));

        var serviceSettings = (AlibabaCloudSearchSparseServiceSettings) model.getServiceSettings();
        assertCommonModelFields(model, serviceSettings.getCommonSettings(), modelIncludesSecrets, minimalSettings);

        var taskSettings = (AlibabaCloudSearchSparseTaskSettings) model.getTaskSettings();
        if (minimalSettings) {
            // Check default values
            assertThat(taskSettings.getInputType(), nullValue());
            assertThat(taskSettings.isReturnToken(), nullValue());
        } else {
            // Check configured values
            assertThat(taskSettings.getInputType(), is(InputType.fromString(INPUT_TYPE_VALUE)));
            assertThat(taskSettings.isReturnToken(), is(RETURN_TOKEN_VALUE));
        }
    }

    private static void assertRerankModel(AlibabaCloudSearchModel model, boolean modelIncludesSecrets, boolean minimalSettings) {
        assertThat(model.getTaskType(), is(TaskType.RERANK));

        var serviceSettings = (AlibabaCloudSearchRerankServiceSettings) model.getServiceSettings();
        assertCommonModelFields(model, serviceSettings.getCommonSettings(), modelIncludesSecrets, minimalSettings);
    }

    private static void assertCompletionModel(AlibabaCloudSearchModel model, boolean modelIncludesSecrets, boolean minimalSettings) {
        assertThat(model.getTaskType(), is(TaskType.COMPLETION));

        var serviceSettings = (AlibabaCloudSearchCompletionServiceSettings) model.getServiceSettings();
        assertCommonModelFields(model, serviceSettings.getCommonSettings(), modelIncludesSecrets, minimalSettings);
    }
}
