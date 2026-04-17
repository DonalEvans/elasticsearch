/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.azureopenai;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.core.Strings;
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
import org.elasticsearch.xpack.inference.services.ServiceFields;
import org.elasticsearch.xpack.inference.services.azureopenai.completion.AzureOpenAiCompletionModel;
import org.elasticsearch.xpack.inference.services.azureopenai.completion.AzureOpenAiCompletionServiceSettings;
import org.elasticsearch.xpack.inference.services.azureopenai.completion.AzureOpenAiCompletionTaskSettings;
import org.elasticsearch.xpack.inference.services.azureopenai.embeddings.AzureOpenAiEmbeddingsModel;
import org.elasticsearch.xpack.inference.services.azureopenai.embeddings.AzureOpenAiEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.azureopenai.embeddings.AzureOpenAiEmbeddingsTaskSettings;
import org.elasticsearch.xpack.inference.services.azureopenai.secrets.AzureOpenAiEntraIdApiKeySecrets;
import org.elasticsearch.xpack.inference.services.azureopenai.secrets.AzureOpenAiOAuth2Secrets;
import org.elasticsearch.xpack.inference.services.azureopenai.secrets.AzureOpenAiSecretSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettingsTests;
import org.junit.Assert;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.test.ESTestCase.assertThat;
import static org.elasticsearch.xpack.inference.Utils.mockClusterServiceEmpty;
import static org.elasticsearch.xpack.inference.common.oauth2.OAuth2Secrets.CLIENT_SECRET_FIELD;
import static org.elasticsearch.xpack.inference.common.oauth2.OAuth2Settings.CLIENT_ID_FIELD;
import static org.elasticsearch.xpack.inference.common.oauth2.OAuth2Settings.SCOPES_FIELD;
import static org.elasticsearch.xpack.inference.common.parser.Headers.HEADERS_FIELD;
import static org.elasticsearch.xpack.inference.services.ConfigurationParseContext.PERSISTENT;
import static org.elasticsearch.xpack.inference.services.ServiceComponentsTests.createWithEmptySettings;
import static org.elasticsearch.xpack.inference.services.ServiceFields.DIMENSIONS_SET_BY_USER;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MAX_INPUT_TOKENS;
import static org.elasticsearch.xpack.inference.services.azureopenai.AzureOpenAiOAuth2Settings.TENANT_ID_FIELD;
import static org.elasticsearch.xpack.inference.services.azureopenai.AzureOpenAiServiceFields.API_VERSION;
import static org.elasticsearch.xpack.inference.services.azureopenai.AzureOpenAiServiceFields.DEPLOYMENT_ID;
import static org.elasticsearch.xpack.inference.services.azureopenai.AzureOpenAiServiceFields.RESOURCE_NAME;
import static org.elasticsearch.xpack.inference.services.azureopenai.AzureOpenAiServiceFields.USER;
import static org.elasticsearch.xpack.inference.services.azureopenai.secrets.AzureOpenAiSecretSettings.API_KEY;
import static org.elasticsearch.xpack.inference.services.azureopenai.secrets.AzureOpenAiSecretSettings.ENTRA_ID;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.nullValue;
import static org.hamcrest.Matchers.instanceOf;

public class AzureOpenAiServiceParameterizedTestConfiguration {
    private static final int MAX_INPUT_TOKENS_VALUE = 123;
    private static final SimilarityMeasure SIMILARITY_VALUE = SimilarityMeasure.DOT_PRODUCT;
    private static final int DIMENSIONS_VALUE = 100;
    private static final String USER_VALUE = "user";
    private static final Map<String, String> HEADERS_VALUE = Map.of("header_key", "header_value");
    private static final String SECRET_VALUE = "secret";
    private static final int REQUESTS_PER_MINUTE = 123;
    public static final String RESOURCE_NAME_VALUE = "resourceName";
    public static final String DEPLOYMENT_ID_VALUE = "deploymentId";
    public static final String API_VERSION_VALUE = "2";
    public static final String CLIENT_ID_VALUE = "clientId";
    public static final List<String> SCOPES_VALUE = List.of("scope1, scope2");
    public static final String TENANT_ID_VALUE = "tenantId";
    public static final boolean DIMENSIONS_SET_BY_USER_VALUE = true;

    public enum AzureOpenAiSecretsTypes {
        API_KEY,
        ENTRA_ID,
        OAUTH2
    }

    public static AbstractInferenceServiceBaseTests.TestConfiguration createTestConfiguration(AzureOpenAiSecretsTypes secretsType) {
        return new AbstractInferenceServiceBaseTests.TestConfiguration.Builder(
            new AbstractInferenceServiceBaseTests.CommonConfig(
                TaskType.TEXT_EMBEDDING,
                EnumSet.of(TaskType.TEXT_EMBEDDING, TaskType.COMPLETION, TaskType.CHAT_COMPLETION),
                AzureOpenAiService.NAME
            ) {
                @Override
                protected AzureOpenAiService createService(ThreadPool threadPool, HttpClientManager clientManager) {
                    var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);
                    return new AzureOpenAiService(senderFactory, createWithEmptySettings(threadPool), mockClusterServiceEmpty());
                }

                @Override
                protected Map<String, Object> createMinimalServiceSettingsMap(TaskType taskType) {
                    return createMinimalServiceSettingsMap(taskType, ConfigurationParseContext.REQUEST);
                }

                @Override
                protected Map<String, Object> createMinimalServiceSettingsMap(TaskType taskType, ConfigurationParseContext parseContext) {
                    var settings = new HashMap<String, Object>(
                        Map.of(RESOURCE_NAME, RESOURCE_NAME_VALUE, DEPLOYMENT_ID, DEPLOYMENT_ID_VALUE, API_VERSION, API_VERSION_VALUE)
                    );
                    // Add OAuth2-specific settings
                    if (secretsType == AzureOpenAiSecretsTypes.OAUTH2) {
                        settings.putAll(
                            Map.of(CLIENT_ID_FIELD, CLIENT_ID_VALUE, SCOPES_FIELD, SCOPES_VALUE, TENANT_ID_FIELD, TENANT_ID_VALUE)
                        );
                    }
                    if (taskType == TaskType.TEXT_EMBEDDING && parseContext == PERSISTENT) {
                        settings.put(DIMENSIONS_SET_BY_USER, DIMENSIONS_SET_BY_USER_VALUE);
                    }
                    return settings;
                }

                @Override
                protected Map<String, Object> createAllServiceSettingsMap(TaskType taskType, ConfigurationParseContext parseContext) {
                    var serviceSettings = createMinimalServiceSettingsMap(taskType);
                    RateLimitSettingsTests.addRateLimitSettingsToMap(serviceSettings, REQUESTS_PER_MINUTE);
                    if (taskType == TaskType.TEXT_EMBEDDING) {
                        serviceSettings.putAll(
                            Map.of(
                                ServiceFields.DIMENSIONS,
                                DIMENSIONS_VALUE,
                                ServiceFields.SIMILARITY,
                                SIMILARITY_VALUE.toString(),
                                MAX_INPUT_TOKENS,
                                MAX_INPUT_TOKENS_VALUE
                            )
                        );
                        if (parseContext == PERSISTENT) {
                            serviceSettings.put(DIMENSIONS_SET_BY_USER, true);
                        }
                    }
                    return serviceSettings;
                }

                @Override
                protected ServiceSettings getServiceSettings(
                    Map<String, Object> serviceSettings,
                    TaskType taskType,
                    ConfigurationParseContext context
                ) {
                    return switch (taskType) {
                        case TEXT_EMBEDDING -> AzureOpenAiEmbeddingsServiceSettings.fromMap(serviceSettings, context);
                        case COMPLETION, CHAT_COMPLETION -> AzureOpenAiCompletionServiceSettings.fromMap(serviceSettings, context);
                        default -> throw new IllegalStateException("Unexpected value: " + taskType);
                    };
                }

                @Override
                protected TaskSettings getEmptyTaskSettings(TaskType taskType) {
                    return switch (taskType) {
                        case TEXT_EMBEDDING -> AzureOpenAiEmbeddingsTaskSettings.EMPTY;
                        case COMPLETION, CHAT_COMPLETION -> AzureOpenAiCompletionTaskSettings.EMPTY;
                        default -> throw new IllegalStateException("Unexpected value: " + taskType);
                    };
                }

                @Override
                protected ModelSecrets createModelSecrets() {
                    return new ModelSecrets(AzureOpenAiSecretSettings.fromMap(createSecretSettingsMap()));
                }

                @Override
                protected Map<String, Object> createAllTaskSettingsMap(TaskType taskType) {
                    return new HashMap<>(Map.of(USER, USER_VALUE, HEADERS_FIELD, HEADERS_VALUE));
                }

                @Override
                protected Map<String, Object> createSecretSettingsMap() {
                    return switch (secretsType) {
                        case API_KEY -> new HashMap<>(Map.of(API_KEY, SECRET_VALUE));
                        case ENTRA_ID -> new HashMap<>(Map.of(ENTRA_ID, SECRET_VALUE));
                        case OAUTH2 -> new HashMap<>(Map.of(CLIENT_SECRET_FIELD, SECRET_VALUE));
                    };
                }

                @Override
                protected void assertModel(
                    Model model,
                    TaskType taskType,
                    boolean modelIncludesSecrets,
                    boolean minimalSettings,
                    ConfigurationParseContext parseContext
                ) {
                    switch (taskType) {
                        case TEXT_EMBEDDING -> assertTextEmbeddingModel(
                            model,
                            modelIncludesSecrets,
                            minimalSettings,
                            parseContext,
                            secretsType
                        );
                        case COMPLETION, CHAT_COMPLETION -> assertCompletionModel(
                            model,
                            modelIncludesSecrets,
                            minimalSettings,
                            secretsType
                        );
                        default -> Assert.fail("unexpected task type: " + taskType);
                    }
                }

                @Override
                protected void assertModel(Model model, TaskType taskType, boolean modelIncludesSecrets, boolean minimalSettings) {
                    assertModel(model, taskType, modelIncludesSecrets, minimalSettings, ConfigurationParseContext.REQUEST);
                }

                @Override
                protected EnumSet<TaskType> supportedStreamingTasks() {
                    return EnumSet.of(TaskType.CHAT_COMPLETION, TaskType.COMPLETION);
                }
            }
        ).build();
    }

    private static void assertCommonSettings(
        AzureOpenAiModel model,
        boolean modelIncludesSecrets,
        boolean minimalSettings,
        AzureOpenAiSecretsTypes secretsType
    ) {
        assertThat(model.resourceName(), is(RESOURCE_NAME_VALUE));
        assertThat(model.deploymentId(), is(DEPLOYMENT_ID_VALUE));
        assertThat(model.apiVersion(), is(API_VERSION_VALUE));

        var serviceSettings = (AzureOpenAiServiceSettings) model.getServiceSettings();
        switch (secretsType) {
            case API_KEY, ENTRA_ID -> assertThat(serviceSettings.oAuth2Settings(), is(nullValue()));
            case OAUTH2 -> {
                assertThat(serviceSettings.oAuth2Settings().clientId(), is(CLIENT_ID_VALUE));
                assertThat(serviceSettings.oAuth2Settings().scopes(), is(SCOPES_VALUE));
                assertThat(serviceSettings.oAuth2Settings().tenantId(), is(TENANT_ID_VALUE));
            }
        }

        var taskSettings = (AzureOpenAiTaskSettings<?>) model.getTaskSettings();
        if (minimalSettings) {
            // Check default values
            assertThat(taskSettings.isEmpty(), is(true));
        } else {
            // Check configured values
            assertThat(taskSettings.user().get(), is(USER_VALUE));
            assertThat(taskSettings.headers().mapValue().get(), is(HEADERS_VALUE));
        }

        if (modelIncludesSecrets) {
            switch (secretsType) {
                case API_KEY -> {
                    assertThat(model.getSecretSettings(), instanceOf(AzureOpenAiEntraIdApiKeySecrets.class));
                    assertThat(
                        ((AzureOpenAiEntraIdApiKeySecrets) model.getSecretSettings()).apiKey(),
                        is(new SecureString(SECRET_VALUE.toCharArray()))
                    );
                    assertThat(((AzureOpenAiEntraIdApiKeySecrets) model.getSecretSettings()).entraId(), is(nullValue()));
                }
                case ENTRA_ID -> {
                    assertThat(model.getSecretSettings(), instanceOf(AzureOpenAiEntraIdApiKeySecrets.class));
                    assertThat(((AzureOpenAiEntraIdApiKeySecrets) model.getSecretSettings()).apiKey(), is(nullValue()));
                    assertThat(
                        ((AzureOpenAiEntraIdApiKeySecrets) model.getSecretSettings()).entraId(),
                        is(new SecureString(SECRET_VALUE.toCharArray()))
                    );
                }
                case OAUTH2 -> {
                    assertThat(model.getSecretSettings(), instanceOf(AzureOpenAiOAuth2Secrets.class));
                    assertThat(((AzureOpenAiOAuth2Secrets) model.getSecretSettings()).getClientSecret(), is(SECRET_VALUE));
                }
            }
        } else {
            assertThat(model.getSecretSettings(), is(nullValue()));
        }
    }

    private static void assertTextEmbeddingModel(
        Model model,
        boolean modelIncludesSecrets,
        boolean minimalSettings,
        ConfigurationParseContext parseContext,
        AzureOpenAiSecretsTypes secretsType
    ) {
        assertThat(model, instanceOf(AzureOpenAiEmbeddingsModel.class));
        var embeddingsModel = (AzureOpenAiEmbeddingsModel) model;
        assertCommonSettings(embeddingsModel, modelIncludesSecrets, minimalSettings, secretsType);

        assertThat(
            embeddingsModel.getUri().toString(),
            is(
                Strings.format(
                    "https://%s.openai.azure.com/openai/deployments/%s/embeddings?api-version=%s",
                    RESOURCE_NAME_VALUE,
                    DEPLOYMENT_ID_VALUE,
                    API_VERSION_VALUE
                )
            )
        );
        if (minimalSettings) {
            if (parseContext == PERSISTENT) {
                assertThat(embeddingsModel.getServiceSettings().dimensionsSetByUser(), is(DIMENSIONS_SET_BY_USER_VALUE));
            } else {
                assertThat(embeddingsModel.getServiceSettings().dimensionsSetByUser(), is(false));
            }
        } else {
            assertThat(embeddingsModel.getServiceSettings().dimensionsSetByUser(), is(DIMENSIONS_SET_BY_USER_VALUE));
        }

        if (minimalSettings) {
            // Check default values
            assertThat(embeddingsModel.getServiceSettings().maxInputTokens(), is(nullValue()));
            assertThat(embeddingsModel.getServiceSettings().dimensions(), is(nullValue()));
            assertThat(embeddingsModel.getServiceSettings().similarity(), is(nullValue()));
            assertThat(embeddingsModel.getServiceSettings().rateLimitSettings(), is(new RateLimitSettings(1440)));
        } else {
            // Check configured values
            assertThat(embeddingsModel.getServiceSettings().maxInputTokens(), is(MAX_INPUT_TOKENS_VALUE));
            assertThat(embeddingsModel.getServiceSettings().dimensions(), is(DIMENSIONS_VALUE));
            assertThat(embeddingsModel.getServiceSettings().similarity(), is(SIMILARITY_VALUE));
            assertThat(embeddingsModel.getServiceSettings().rateLimitSettings(), is(new RateLimitSettings(REQUESTS_PER_MINUTE)));
        }
    }

    private static void assertCompletionModel(
        Model model,
        boolean modelIncludesSecrets,
        boolean minimalSettings,
        AzureOpenAiSecretsTypes secretsType
    ) {
        assertThat(model, instanceOf(AzureOpenAiCompletionModel.class));
        var completionModel = (AzureOpenAiCompletionModel) model;

        assertCommonSettings(completionModel, modelIncludesSecrets, minimalSettings, secretsType);

        assertThat(
            completionModel.getUri().toString(),
            is(
                Strings.format(
                    "https://%s.openai.azure.com/openai/deployments/%s/chat/completions?api-version=%s",
                    RESOURCE_NAME_VALUE,
                    DEPLOYMENT_ID_VALUE,
                    API_VERSION_VALUE
                )
            )
        );
        assertThat(completionModel.getServiceSettings().dimensions(), is(nullValue()));
        assertThat(completionModel.getServiceSettings().similarity(), is(nullValue()));
        if (minimalSettings) {
            // Check default values
            assertThat(completionModel.getServiceSettings().rateLimitSettings(), is(new RateLimitSettings(120)));
        } else {
            // Check configured values
            assertThat(completionModel.getServiceSettings().rateLimitSettings(), is(new RateLimitSettings(REQUESTS_PER_MINUTE)));
        }
    }

}
