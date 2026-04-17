/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.amazonbedrock;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.ModelSecrets;
import org.elasticsearch.inference.ServiceSettings;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskSettings;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.inference.common.amazon.AwsSecretSettings;
import org.elasticsearch.xpack.inference.common.model.Truncation;
import org.elasticsearch.xpack.inference.external.http.HttpClientManager;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSenderTests;
import org.elasticsearch.xpack.inference.services.AbstractInferenceServiceBaseTests;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.ServiceComponentsTests;
import org.elasticsearch.xpack.inference.services.amazonbedrock.client.AmazonBedrockMockRequestSender;
import org.elasticsearch.xpack.inference.services.amazonbedrock.completion.AmazonBedrockChatCompletionServiceSettings;
import org.elasticsearch.xpack.inference.services.amazonbedrock.completion.AmazonBedrockCompletionTaskSettings;
import org.elasticsearch.xpack.inference.services.amazonbedrock.embeddings.AmazonBedrockEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.amazonbedrock.embeddings.AmazonBedrockEmbeddingsTaskSettings;
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
import static org.elasticsearch.xpack.inference.services.ServiceFields.DIMENSIONS_SET_BY_USER;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MAX_INPUT_TOKENS;
import static org.elasticsearch.xpack.inference.services.ServiceFields.SIMILARITY;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.ACCESS_KEY_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.MAX_NEW_TOKENS_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.MODEL_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.PROVIDER_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.REGION_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.SECRET_KEY_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.TEMPERATURE_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.TOP_K_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.TOP_P_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockConstants.TRUNCATE_FIELD;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockProvider.ANTHROPIC;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockProvider.COHERE;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.AmazonBedrockProvider.MISTRAL;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.Assert.fail;

public class AmazonBedrockServiceParameterizedTestConfiguration {

    public static final String MODEL_ID_VALUE = "modelId";
    public static final String REGION_VALUE = "someRegion";
    public static final int REQUESTS_PER_MINUTE_VALUE = 123;
    public static final String SIMILARITY_VALUE = SimilarityMeasure.COSINE.toString();
    public static final int DIMENSIONS_VALUE = 456;
    public static final int MAX_INPUT_TOKENS_VALUE = 789;
    public static final String ACCESS_KEY_VALUE = "accessKey";
    public static final String SECRET_KEY_VALUE = "secretKey";
    public static final double TEMPERATURE_VALUE = 0.5;
    public static final double TOP_P_VALUE = 0.1;
    public static final double TOP_K_VALUE = 0.2;
    public static final int MAX_NEW_TOKENS_VALUE = 135;
    public static final Truncation TRUNCATION_VALUE = Truncation.START;
    public static final boolean DIMENSIONS_SET_BY_USER_VALUE = true;

    public static AbstractInferenceServiceBaseTests.TestConfiguration createTestConfiguration(AmazonBedrockProvider provider) {
        EnumSet<TaskType> supportedTasks = switch (provider) {
            case AMAZONTITAN, COHERE -> EnumSet.of(TaskType.TEXT_EMBEDDING, TaskType.COMPLETION, TaskType.CHAT_COMPLETION);
            case ANTHROPIC, AI21LABS, META, MISTRAL -> EnumSet.of(TaskType.COMPLETION, TaskType.CHAT_COMPLETION);
        };
        return new AbstractInferenceServiceBaseTests.TestConfiguration.Builder(
            new AbstractInferenceServiceBaseTests.CommonConfig(randomFrom(supportedTasks), supportedTasks, AmazonBedrockService.NAME) {

                @Override
                protected AmazonBedrockService createService(ThreadPool threadPool, HttpClientManager clientManager) {
                    var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
                        ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
                        mockClusterServiceEmpty()
                    );
                    return new AmazonBedrockService(
                        HttpRequestSenderTests.createSenderFactory(threadPool, clientManager),
                        amazonBedrockFactory,
                        createWithEmptySettings(threadPool),
                        mockClusterServiceEmpty()
                    );
                }

                @Override
                protected Map<String, Object> createMinimalServiceSettingsMap(TaskType taskType) {
                    return createMinimalServiceSettingsMap(taskType, ConfigurationParseContext.REQUEST);
                }

                @Override
                protected Map<String, Object> createMinimalServiceSettingsMap(TaskType taskType, ConfigurationParseContext parseContext) {
                    var settings = new HashMap<String, Object>(
                        Map.of(MODEL_FIELD, MODEL_ID_VALUE, REGION_FIELD, REGION_VALUE, PROVIDER_FIELD, provider.toString())
                    );
                    if (taskType == TaskType.TEXT_EMBEDDING && parseContext == ConfigurationParseContext.PERSISTENT) {
                        settings.put(DIMENSIONS_SET_BY_USER, DIMENSIONS_SET_BY_USER_VALUE);
                    }
                    return settings;
                }

                @Override
                protected Map<String, Object> createAllSupportedServiceSettingsMap(
                    TaskType taskType,
                    ConfigurationParseContext parseContext
                ) {
                    var serviceSettings = createMinimalServiceSettingsMap(taskType);
                    RateLimitSettingsTests.addRateLimitSettingsToMap(serviceSettings, REQUESTS_PER_MINUTE_VALUE);
                    if (taskType.equals(TaskType.TEXT_EMBEDDING)) {
                        serviceSettings.put(SIMILARITY, SIMILARITY_VALUE);
                        serviceSettings.put(MAX_INPUT_TOKENS, MAX_INPUT_TOKENS_VALUE);
                        if (parseContext == ConfigurationParseContext.PERSISTENT) {
                            serviceSettings.put(DIMENSIONS, DIMENSIONS_VALUE);
                        }
                    }
                    return serviceSettings;
                }

                @Override
                protected ModelSecrets createModelSecrets() {
                    return new ModelSecrets(AwsSecretSettings.fromMap(createSecretSettingsMap()));
                }

                @Override
                protected ServiceSettings getServiceSettings(
                    Map<String, Object> serviceSettings,
                    TaskType taskType,
                    ConfigurationParseContext context
                ) {
                    return switch (taskType) {
                        case TEXT_EMBEDDING -> AmazonBedrockEmbeddingsServiceSettings.fromMap(serviceSettings, context);
                        case COMPLETION, CHAT_COMPLETION -> AmazonBedrockChatCompletionServiceSettings.fromMap(serviceSettings, context);
                        default -> throw new IllegalArgumentException("Unsupported task type: " + taskType);
                    };
                }

                @Override
                protected TaskSettings getEmptyTaskSettings(TaskType taskType) {
                    return switch (taskType) {
                        case TEXT_EMBEDDING -> AmazonBedrockEmbeddingsTaskSettings.fromMap(null);
                        case COMPLETION, CHAT_COMPLETION -> AmazonBedrockCompletionTaskSettings.fromMap(null);
                        default -> throw new IllegalArgumentException("Unsupported task type: " + taskType);
                    };
                }

                @Override
                protected Map<String, Object> createTaskSettingsMap(TaskType taskType) {
                    var taskSettingsMap = new HashMap<String, Object>();
                    if (taskType.equals(TaskType.TEXT_EMBEDDING) && provider.equals(COHERE)) {
                        taskSettingsMap.put(TRUNCATE_FIELD, TRUNCATION_VALUE.toString());
                    } else if (taskType.equals(TaskType.COMPLETION) || taskType.equals(TaskType.CHAT_COMPLETION)) {
                        taskSettingsMap.putAll(
                            Map.of(
                                TEMPERATURE_FIELD,
                                TEMPERATURE_VALUE,
                                TOP_P_FIELD,
                                TOP_P_VALUE,
                                MAX_NEW_TOKENS_FIELD,
                                MAX_NEW_TOKENS_VALUE
                            )
                        );
                        if (EnumSet.of(ANTHROPIC, COHERE, MISTRAL).contains(provider)) {
                            taskSettingsMap.put(TOP_K_FIELD, TOP_K_VALUE);
                        }
                    }
                    return taskSettingsMap;
                }

                @Override
                protected Map<String, Object> createSecretSettingsMap() {
                    return new HashMap<>(Map.of(ACCESS_KEY_FIELD, ACCESS_KEY_VALUE, SECRET_KEY_FIELD, SECRET_KEY_VALUE));
                }

                @Override
                protected void assertModel(Model model, TaskType taskType, boolean modelIncludesSecrets, boolean minimalSettings) {
                    assertModel(model, taskType, modelIncludesSecrets, minimalSettings, ConfigurationParseContext.REQUEST);
                }

                @Override
                protected void assertModel(
                    Model model,
                    TaskType taskType,
                    boolean modelIncludesSecrets,
                    boolean minimalSettings,
                    ConfigurationParseContext parseContext
                ) {
                    assertThat(model, instanceOf(AmazonBedrockModel.class));
                    var amazonModel = (AmazonBedrockModel) model;
                    switch (taskType) {
                        case TEXT_EMBEDDING -> assertTextEmbeddingModel(
                            amazonModel,
                            modelIncludesSecrets,
                            minimalSettings,
                            parseContext,
                            provider
                        );
                        case COMPLETION, CHAT_COMPLETION -> assertCompletionModel(
                            amazonModel,
                            taskType,
                            modelIncludesSecrets,
                            minimalSettings,
                            provider
                        );
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
        AmazonBedrockModel amazonModel,
        boolean modelIncludesSecrets,
        boolean minimalSettings,
        AmazonBedrockProvider provider
    ) {
        assertThat(amazonModel.getServiceSettings().modelId(), is(MODEL_ID_VALUE));
        assertThat(amazonModel.getServiceSettings().region(), is(REGION_VALUE));
        assertThat(amazonModel.getServiceSettings().provider(), is(provider));

        if (minimalSettings) {
            assertThat(amazonModel.getServiceSettings().rateLimitSettings(), is(new RateLimitSettings(240)));
        } else {
            assertThat(amazonModel.getServiceSettings().rateLimitSettings(), is(new RateLimitSettings(REQUESTS_PER_MINUTE_VALUE)));
        }

        if (modelIncludesSecrets) {
            assertThat(amazonModel.getSecretSettings().accessKey(), is(new SecureString(ACCESS_KEY_VALUE.toCharArray())));
            assertThat(amazonModel.getSecretSettings().secretKey(), is(new SecureString(SECRET_KEY_VALUE.toCharArray())));
        } else {
            assertThat(amazonModel.getSecretSettings(), is(nullValue()));
        }
    }

    private static void assertTextEmbeddingModel(
        AmazonBedrockModel model,
        boolean modelIncludesSecrets,
        boolean minimalSettings,
        ConfigurationParseContext parseContext,
        AmazonBedrockProvider provider
    ) {
        assertThat(model.getTaskType(), is(TaskType.TEXT_EMBEDDING));
        assertCommonModelFields(model, modelIncludesSecrets, minimalSettings, provider);

        var serviceSettings = (AmazonBedrockEmbeddingsServiceSettings) model.getServiceSettings();

        if (parseContext == ConfigurationParseContext.PERSISTENT) {
            // DIMENSIONS_SET_BY_USER is required for PERSISTENT context
            assertThat(serviceSettings.dimensionsSetByUser(), is(DIMENSIONS_SET_BY_USER_VALUE));
            if (minimalSettings == false) {
                assertThat(serviceSettings.dimensions(), is(DIMENSIONS_VALUE));
            }
        } else {
            // DIMENSIONS is not supported for REQUEST context
            assertThat(serviceSettings.dimensions(), is(nullValue()));
            assertThat(serviceSettings.dimensionsSetByUser(), is(false));
        }

        var taskSettings = (AmazonBedrockEmbeddingsTaskSettings) model.getTaskSettings();
        if (minimalSettings) {
            // Check default values
            assertThat(serviceSettings.similarity(), is(nullValue()));
            assertThat(serviceSettings.maxInputTokens(), is(nullValue()));
            assertThat(taskSettings.isEmpty(), is(true));
        } else {
            // Check configured values
            assertThat(serviceSettings.similarity(), is(SimilarityMeasure.fromString(SIMILARITY_VALUE)));
            assertThat(serviceSettings.maxInputTokens(), is(MAX_INPUT_TOKENS_VALUE));
            if (provider.equals(COHERE)) {
                assertThat(taskSettings.truncation(), is(TRUNCATION_VALUE));
            } else {
                assertThat(taskSettings.isEmpty(), is(true));
            }
        }
    }

    private static void assertCompletionModel(
        AmazonBedrockModel model,
        TaskType taskType,
        boolean modelIncludesSecrets,
        boolean minimalSettings,
        AmazonBedrockProvider provider
    ) {
        assertThat(model.getTaskType(), is(taskType));
        assertCommonModelFields(model, modelIncludesSecrets, minimalSettings, provider);

        var taskSettings = (AmazonBedrockCompletionTaskSettings) model.getTaskSettings();
        if (minimalSettings) {
            // Check default values
            assertThat(taskSettings.isEmpty(), is(true));
        } else {
            // Check configured values
            assertThat(taskSettings.temperature(), is(TEMPERATURE_VALUE));
            assertThat(taskSettings.topP(), is(TOP_P_VALUE));
            assertThat(taskSettings.maxNewTokens(), is(MAX_NEW_TOKENS_VALUE));
            if (EnumSet.of(ANTHROPIC, COHERE, MISTRAL).contains(provider)) {
                assertThat(taskSettings.topK(), is(TOP_K_VALUE));
            }
        }
    }
}
