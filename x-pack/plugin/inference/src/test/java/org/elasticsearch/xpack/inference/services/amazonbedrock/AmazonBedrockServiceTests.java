/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.amazonbedrock;

import software.amazon.awssdk.services.bedrockruntime.model.BedrockRuntimeException;

import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.support.ActionTestUtils;
import org.elasticsearch.action.support.PlainActionFuture;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.inference.ChunkInferenceInput;
import org.elasticsearch.inference.ChunkedInference;
import org.elasticsearch.inference.InferenceService;
import org.elasticsearch.inference.InferenceServiceConfiguration;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.inference.UnifiedCompletionRequest;
import org.elasticsearch.inference.completion.ContentString;
import org.elasticsearch.inference.completion.Message;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.core.inference.action.InferenceAction;
import org.elasticsearch.xpack.core.inference.results.ChatCompletionResults;
import org.elasticsearch.xpack.core.inference.results.ChunkedInferenceEmbedding;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingFloatResults;
import org.elasticsearch.xpack.inference.common.model.Truncation;
import org.elasticsearch.xpack.inference.external.http.HttpClientManager;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSender;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSenderTests;
import org.elasticsearch.xpack.inference.logging.ThrottlerManager;
import org.elasticsearch.xpack.inference.services.InferenceServiceTestCase;
import org.elasticsearch.xpack.inference.services.ServiceComponentsTests;
import org.elasticsearch.xpack.inference.services.amazonbedrock.client.AmazonBedrockMockRequestSender;
import org.elasticsearch.xpack.inference.services.amazonbedrock.completion.AmazonBedrockChatCompletionModelTests;
import org.elasticsearch.xpack.inference.services.amazonbedrock.embeddings.AmazonBedrockEmbeddingsModel;
import org.elasticsearch.xpack.inference.services.amazonbedrock.embeddings.AmazonBedrockEmbeddingsModelTests;
import org.elasticsearch.xpack.inference.services.amazonbedrock.embeddings.AmazonBedrockEmbeddingsTaskSettingsTests;
import org.hamcrest.CoreMatchers;
import org.hamcrest.Matchers;
import org.junit.After;
import org.junit.Before;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.elasticsearch.common.xcontent.XContentHelper.toXContent;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertToXContentEquivalent;
import static org.elasticsearch.xpack.core.inference.chunking.ChunkingSettingsTests.createRandomChunkingSettings;
import static org.elasticsearch.xpack.core.inference.results.ChatCompletionResultsTests.buildExpectationCompletion;
import static org.elasticsearch.xpack.core.inference.results.DenseEmbeddingFloatResultsTests.buildExpectationFloat;
import static org.elasticsearch.xpack.inference.Utils.inferenceUtilityExecutors;
import static org.elasticsearch.xpack.inference.Utils.mockClusterServiceEmpty;
import static org.elasticsearch.xpack.inference.common.amazon.AwsSecretSettingsTests.getAmazonBedrockSecretSettingsMap;
import static org.elasticsearch.xpack.inference.services.SenderServiceTests.createMockSender;
import static org.elasticsearch.xpack.inference.services.ServiceComponentsTests.createWithEmptySettings;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.completion.AmazonBedrockChatCompletionServiceSettingsTests.createChatCompletionRequestSettingsMap;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.completion.AmazonBedrockCompletionTaskSettingsTests.getChatCompletionTaskSettingsMap;
import static org.elasticsearch.xpack.inference.services.amazonbedrock.embeddings.AmazonBedrockEmbeddingsServiceSettingsTests.createEmbeddingsRequestSettingsMap;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class AmazonBedrockServiceTests extends InferenceServiceTestCase {
    private static final TimeValue TIMEOUT = new TimeValue(30, TimeUnit.SECONDS);
    private static final String INFERENCE_ID_VALUE = "id";
    private static final String REGION_VALUE = "region";
    private static final String MODEL_VALUE = "model";
    private static final AmazonBedrockProvider AMAZON_BEDROCK_PROVIDER_VALUE = AmazonBedrockProvider.AMAZONTITAN;
    private static final String ACCESS_KEY_VALUE = "access";
    private static final String SECRET_KEY_VALUE = "secret";
    private ThreadPool threadPool;
    private HttpClientManager clientManager;

    @Before
    public void init() throws Exception {
        threadPool = createThreadPool(inferenceUtilityExecutors());
        clientManager = HttpClientManager.create(Settings.EMPTY, threadPool, mockClusterServiceEmpty(), mock(ThrottlerManager.class));
    }

    @After
    public void shutdown() throws IOException {
        clientManager.close();
        terminate(threadPool);
    }

    public void testParseRequestConfig_CohereSettingsWithNoCohereModel() throws IOException {
        try (var service = createAmazonBedrockService()) {
            ActionListener<Model> modelVerificationListener = ActionTestUtils.assertNoSuccessListener(exception -> {
                assertThat(exception, instanceOf(ElasticsearchStatusException.class));
                assertThat(
                    exception.getMessage(),
                    is("The [text_embedding] task type for provider [amazontitan] does not allow [truncate] field")
                );
            });

            service.parseRequestConfig(
                INFERENCE_ID_VALUE,
                TaskType.TEXT_EMBEDDING,
                getRequestConfigMap(
                    createEmbeddingsRequestSettingsMap(REGION_VALUE, MODEL_VALUE, "amazontitan", null, null, null, null),
                    AmazonBedrockEmbeddingsTaskSettingsTests.mutableMap("truncate", Truncation.START),
                    getAmazonBedrockSecretSettingsMap(ACCESS_KEY_VALUE, SECRET_KEY_VALUE)
                ),
                modelVerificationListener
            );
        }
    }

    @SuppressWarnings("checkstyle:LineLength")
    public void testGetConfiguration() throws Exception {
        try (var service = createAmazonBedrockService()) {
            String content = XContentHelper.stripWhitespace(
                """
                    {
                         "service": "amazonbedrock",
                         "name": "Amazon Bedrock",
                         "task_types": ["text_embedding", "completion", "chat_completion"],
                         "configurations": {
                              "dimensions": {
                                 "description": "The number of dimensions the resulting embeddings should have. For more information refer to https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html.",
                                 "label": "Dimensions",
                                 "required": false,
                                 "sensitive": false,
                                 "updatable": false,
                                 "type": "int",
                                 "supported_task_types": ["text_embedding"]
                             },
                             "secret_key": {
                                 "description": "A valid AWS secret key that is paired with the access_key.",
                                 "label": "Secret Key",
                                 "required": true,
                                 "sensitive": true,
                                 "updatable": true,
                                 "type": "str",
                                 "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                             },
                             "provider": {
                                 "description": "The model provider for your deployment.",
                                 "label": "Provider",
                                 "required": true,
                                 "sensitive": false,
                                 "updatable": false,
                                 "type": "str",
                                 "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                             },
                             "access_key": {
                                 "description": "A valid AWS access key that has permissions to use Amazon Bedrock.",
                                 "label": "Access Key",
                                 "required": true,
                                 "sensitive": true,
                                 "updatable": true,
                                 "type": "str",
                                 "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                             },
                             "model": {
                                 "description": "The base model ID or an ARN to a custom model based on a foundational model.",
                                 "label": "Model",
                                 "required": true,
                                 "sensitive": false,
                                 "updatable": false,
                                 "type": "str",
                                 "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                             },
                             "rate_limit.requests_per_minute": {
                                 "description": "By default, the amazonbedrock service sets the number of requests allowed per minute to 240.",
                                 "label": "Rate Limit",
                                 "required": false,
                                 "sensitive": false,
                                 "updatable": false,
                                 "type": "int",
                                 "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                             },
                             "region": {
                                 "description": "The region that your model or ARN is deployed in.",
                                 "label": "Region",
                                 "required": true,
                                 "sensitive": false,
                                 "updatable": false,
                                 "type": "str",
                                 "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                             }
                         }
                     }
                    """
            );
            InferenceServiceConfiguration configuration = InferenceServiceConfiguration.fromXContentBytes(
                new BytesArray(content),
                XContentType.JSON
            );
            boolean humanReadable = true;
            BytesReference originalBytes = toShuffledXContent(configuration, XContentType.JSON, ToXContent.EMPTY_PARAMS, humanReadable);
            InferenceServiceConfiguration serviceConfiguration = service.getConfiguration();
            assertToXContentEquivalent(
                originalBytes,
                toXContent(serviceConfiguration, XContentType.JSON, humanReadable),
                XContentType.JSON
            );
        }
    }

    public void testParseRequestConfig_ForEmbeddingsTask_InvalidProvider() throws IOException {
        try (var service = createAmazonBedrockService()) {
            ActionListener<Model> modelVerificationListener = ActionTestUtils.assertNoSuccessListener(exception -> {
                assertThat(exception, instanceOf(ElasticsearchStatusException.class));
                assertThat(exception.getMessage(), is("The [text_embedding] task type for provider [anthropic] is not available"));
            });

            service.parseRequestConfig(
                INFERENCE_ID_VALUE,
                TaskType.TEXT_EMBEDDING,
                getRequestConfigMap(
                    createEmbeddingsRequestSettingsMap(REGION_VALUE, MODEL_VALUE, "anthropic", null, null, null, null),
                    Map.of(),
                    getAmazonBedrockSecretSettingsMap(ACCESS_KEY_VALUE, SECRET_KEY_VALUE)
                ),
                modelVerificationListener
            );
        }
    }

    public void testCreateModel_TopKParameter_NotAvailable() throws IOException {
        try (var service = createAmazonBedrockService()) {
            var topKNotAvailableListener = ActionTestUtils.<Model>assertNoSuccessListener(exception -> {
                assertThat(exception, instanceOf(ElasticsearchStatusException.class));
                assertThat(exception.getMessage(), is("The [top_k] task parameter is not available for provider [amazontitan]"));
            });
            assertTopKParameter(TaskType.COMPLETION, service, topKNotAvailableListener);
            assertTopKParameter(TaskType.CHAT_COMPLETION, service, topKNotAvailableListener);
        }
    }

    private void assertTopKParameter(TaskType taskType, AmazonBedrockService service, ActionListener<Model> modelVerificationListener) {
        service.parseRequestConfig(
            INFERENCE_ID_VALUE,
            taskType,
            getRequestConfigMap(
                createChatCompletionRequestSettingsMap(REGION_VALUE, MODEL_VALUE, "amazontitan"),
                getChatCompletionTaskSettingsMap(1.0, 0.5, 0.2, 128),
                getAmazonBedrockSecretSettingsMap(ACCESS_KEY_VALUE, SECRET_KEY_VALUE)
            ),
            modelVerificationListener
        );
    }

    public void testParseRequestConfig_ForEmbeddingsTask_DimensionsIsNotAllowed() throws IOException {
        try (var service = createAmazonBedrockService()) {
            ActionListener<Model> modelVerificationListener = ActionTestUtils.assertNoSuccessListener(exception -> {
                assertThat(exception, instanceOf(ValidationException.class));
                assertThat(exception.getMessage(), containsString("[service_settings] does not allow the setting [dimensions]"));
            });

            service.parseRequestConfig(
                INFERENCE_ID_VALUE,
                TaskType.TEXT_EMBEDDING,
                getRequestConfigMap(
                    createEmbeddingsRequestSettingsMap(REGION_VALUE, MODEL_VALUE, "amazontitan", 512, null, null, null),
                    Map.of(),
                    getAmazonBedrockSecretSettingsMap(ACCESS_KEY_VALUE, SECRET_KEY_VALUE)
                ),
                modelVerificationListener
            );
        }
    }

    public void testInfer_SendsRequest_ForTitanEmbeddingsModel() throws IOException {
        var sender = createMockSender();
        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
            ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
            mockClusterServiceEmpty()
        );
        var model = AmazonBedrockEmbeddingsModelTests.createModel(
            INFERENCE_ID_VALUE,
            REGION_VALUE,
            MODEL_VALUE,
            AMAZON_BEDROCK_PROVIDER_VALUE,
            ACCESS_KEY_VALUE,
            SECRET_KEY_VALUE
        );

        try (
            var service = new AmazonBedrockService(
                factory,
                amazonBedrockFactory,
                createWithEmptySettings(threadPool),
                mockClusterServiceEmpty()
            );
            var requestSender = (AmazonBedrockMockRequestSender) amazonBedrockFactory.createSender()
        ) {
            var results = new DenseEmbeddingFloatResults(List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 0.123F, 0.678F })));
            requestSender.enqueue(results);
            PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
            service.infer(
                model,
                null,
                null,
                null,
                List.of("abc"),
                false,
                new HashMap<>(),
                InputType.INGEST,
                InferenceAction.Request.DEFAULT_TIMEOUT,
                listener
            );

            var result = listener.actionGet(TIMEOUT);

            assertThat(result.asMap(), Matchers.is(buildExpectationFloat(List.of(new float[] { 0.123F, 0.678F }))));
        }
    }

    public void testInfer_SendsRequest_ForCohereEmbeddingsModel() throws IOException {
        var sender = createMockSender();
        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
            ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
            mockClusterServiceEmpty()
        );

        try (
            var service = new AmazonBedrockService(
                factory,
                amazonBedrockFactory,
                createWithEmptySettings(threadPool),
                mockClusterServiceEmpty()
            )
        ) {
            try (var requestSender = (AmazonBedrockMockRequestSender) amazonBedrockFactory.createSender()) {
                var results = new DenseEmbeddingFloatResults(
                    List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 0.123F, 0.678F }))
                );
                requestSender.enqueue(results);

                var model = AmazonBedrockEmbeddingsModelTests.createModel(
                    INFERENCE_ID_VALUE,
                    REGION_VALUE,
                    MODEL_VALUE,
                    AmazonBedrockProvider.COHERE,
                    ACCESS_KEY_VALUE,
                    SECRET_KEY_VALUE
                );
                PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
                service.infer(
                    model,
                    null,
                    null,
                    null,
                    List.of("abc"),
                    false,
                    new HashMap<>(),
                    InputType.CLASSIFICATION,
                    InferenceAction.Request.DEFAULT_TIMEOUT,
                    listener
                );

                var result = listener.actionGet(TIMEOUT);

                assertThat(result.asMap(), Matchers.is(buildExpectationFloat(List.of(new float[] { 0.123F, 0.678F }))));
            }
        }
    }

    public void testInfer_SendsRequest_ForChatCompletionModel() throws IOException {
        var sender = createMockSender();
        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
            ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
            mockClusterServiceEmpty()
        );

        try (
            var service = new AmazonBedrockService(
                factory,
                amazonBedrockFactory,
                createWithEmptySettings(threadPool),
                mockClusterServiceEmpty()
            )
        ) {
            try (var requestSender = (AmazonBedrockMockRequestSender) amazonBedrockFactory.createSender()) {
                var mockResults = new ChatCompletionResults(List.of(new ChatCompletionResults.Result("test result")));
                requestSender.enqueue(mockResults);

                var model = AmazonBedrockChatCompletionModelTests.createCompletionModel(
                    INFERENCE_ID_VALUE,
                    REGION_VALUE,
                    MODEL_VALUE,
                    AMAZON_BEDROCK_PROVIDER_VALUE,
                    ACCESS_KEY_VALUE,
                    SECRET_KEY_VALUE
                );
                PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
                service.infer(
                    model,
                    null,
                    null,
                    null,
                    List.of("abc"),
                    false,
                    new HashMap<>(),
                    InputType.INGEST,
                    InferenceAction.Request.DEFAULT_TIMEOUT,
                    listener
                );

                var result = listener.actionGet(TIMEOUT);

                assertThat(result.asMap(), Matchers.is(buildExpectationCompletion(List.of("test result"))));
            }
        }
    }

    public void testInfer_SendsUnifiedRequest_ForChatCompletionModel() throws IOException {
        var sender = createMockSender();
        var factory = mock(HttpRequestSender.Factory.class);

        when(factory.createSender()).thenReturn(sender);

        var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
            ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
            mockClusterServiceEmpty()
        );

        try (
            var service = new AmazonBedrockService(
                factory,
                amazonBedrockFactory,
                createWithEmptySettings(threadPool),
                mockClusterServiceEmpty()
            )
        ) {
            try (var requestSender = (AmazonBedrockMockRequestSender) amazonBedrockFactory.createSender()) {
                var mockResults = new ChatCompletionResults(List.of(new ChatCompletionResults.Result("test result")));
                requestSender.enqueue(mockResults);

                var model = AmazonBedrockChatCompletionModelTests.createChatCompletionModel(
                    INFERENCE_ID_VALUE,
                    REGION_VALUE,
                    MODEL_VALUE,
                    AmazonBedrockProvider.AMAZONTITAN,
                    ACCESS_KEY_VALUE,
                    SECRET_KEY_VALUE
                );
                PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
                service.unifiedCompletionInfer(
                    model,
                    UnifiedCompletionRequest.of(List.of(new Message(new ContentString("hello"), "user", null, null))),
                    TIMEOUT,
                    listener
                );

                var result = listener.actionGet(TIMEOUT);

                assertThat(result.asMap(), Matchers.is(buildExpectationCompletion(List.of("test result"))));
            }
        }
    }

    public void testInfer_UnauthorizedResponse() throws IOException {
        var sender = createMockSender();
        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
            ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
            mockClusterServiceEmpty()
        );

        try (
            var service = new AmazonBedrockService(
                factory,
                amazonBedrockFactory,
                createWithEmptySettings(threadPool),
                mockClusterServiceEmpty()
            );
            var requestSender = (AmazonBedrockMockRequestSender) amazonBedrockFactory.createSender()
        ) {
            requestSender.enqueue(
                BedrockRuntimeException.builder().message("The security token included in the request is invalid").build()
            );

            var model = AmazonBedrockEmbeddingsModelTests.createModel(
                INFERENCE_ID_VALUE,
                "us-east-1",
                "amazon.titan-embed-text-v1",
                AMAZON_BEDROCK_PROVIDER_VALUE,
                "_INVALID_AWS_ACCESS_KEY_",
                "_INVALID_AWS_SECRET_KEY_"
            );
            PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
            service.infer(
                model,
                null,
                null,
                null,
                List.of("abc"),
                false,
                new HashMap<>(),
                InputType.INTERNAL_INGEST,
                InferenceAction.Request.DEFAULT_TIMEOUT,
                listener
            );

            var exceptionThrown = assertThrows(ElasticsearchException.class, () -> listener.actionGet(TIMEOUT));
            assertThat(exceptionThrown.getCause().getMessage(), containsString("The security token included in the request is invalid"));
        }
    }

    public void testChunkedInfer_ChunkingSettingsSet() throws IOException {
        var model = AmazonBedrockEmbeddingsModelTests.createModel(
            INFERENCE_ID_VALUE,
            REGION_VALUE,
            MODEL_VALUE,
            AMAZON_BEDROCK_PROVIDER_VALUE,
            createRandomChunkingSettings(),
            ACCESS_KEY_VALUE,
            SECRET_KEY_VALUE
        );

        testChunkedInfer(model);
    }

    public void testChunkedInfer_ChunkingSettingsNotSet() throws IOException {
        var model = AmazonBedrockEmbeddingsModelTests.createModel(
            INFERENCE_ID_VALUE,
            REGION_VALUE,
            MODEL_VALUE,
            AMAZON_BEDROCK_PROVIDER_VALUE,
            null,
            ACCESS_KEY_VALUE,
            SECRET_KEY_VALUE
        );

        testChunkedInfer(model);
    }

    public void testChunkedInfer_noInputs() throws IOException {
        var model = AmazonBedrockEmbeddingsModelTests.createModel(
            INFERENCE_ID_VALUE,
            REGION_VALUE,
            MODEL_VALUE,
            AMAZON_BEDROCK_PROVIDER_VALUE,
            null,
            ACCESS_KEY_VALUE,
            SECRET_KEY_VALUE
        );

        var sender = createMockSender();
        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
            ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
            mockClusterServiceEmpty()
        );

        try (
            var service = new AmazonBedrockService(
                factory,
                amazonBedrockFactory,
                createWithEmptySettings(threadPool),
                mockClusterServiceEmpty()
            )
        ) {
            PlainActionFuture<List<ChunkedInference>> listener = new PlainActionFuture<>();
            service.chunkedInfer(
                model,
                null,
                List.of(),
                new HashMap<>(),
                InputType.INTERNAL_INGEST,
                InferenceAction.Request.DEFAULT_TIMEOUT,
                listener
            );

            var results = listener.actionGet(TIMEOUT);
            assertThat(results, empty());
        }
    }

    private void testChunkedInfer(AmazonBedrockEmbeddingsModel model) throws IOException {
        var sender = createMockSender();
        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var amazonBedrockFactory = new AmazonBedrockMockRequestSender.Factory(
            ServiceComponentsTests.createWithSettings(threadPool, Settings.EMPTY),
            mockClusterServiceEmpty()
        );

        try (
            var service = new AmazonBedrockService(
                factory,
                amazonBedrockFactory,
                createWithEmptySettings(threadPool),
                mockClusterServiceEmpty()
            )
        ) {
            try (var requestSender = (AmazonBedrockMockRequestSender) amazonBedrockFactory.createSender()) {
                {
                    var mockResults1 = new DenseEmbeddingFloatResults(
                        List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 0.123F, 0.678F }))
                    );
                    requestSender.enqueue(mockResults1);
                }
                {
                    var mockResults2 = new DenseEmbeddingFloatResults(
                        List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 0.223F, 0.278F }))
                    );
                    requestSender.enqueue(mockResults2);
                }

                PlainActionFuture<List<ChunkedInference>> listener = new PlainActionFuture<>();
                service.chunkedInfer(
                    model,
                    null,
                    List.of(new ChunkInferenceInput("a"), new ChunkInferenceInput("bb")),
                    new HashMap<>(),
                    InputType.INTERNAL_INGEST,
                    InferenceAction.Request.DEFAULT_TIMEOUT,
                    listener
                );

                var results = listener.actionGet(TIMEOUT);
                assertThat(results, hasSize(2));
                {
                    assertThat(results.get(0), CoreMatchers.instanceOf(ChunkedInferenceEmbedding.class));
                    var floatResult = (ChunkedInferenceEmbedding) results.get(0);
                    assertThat(floatResult.chunks(), hasSize(1));
                    assertEquals(new ChunkedInference.TextOffset(0, 1), floatResult.chunks().get(0).offset());
                    assertThat(floatResult.chunks().get(0).embedding(), instanceOf(DenseEmbeddingFloatResults.Embedding.class));
                    assertArrayEquals(
                        new float[] { 0.123F, 0.678F },
                        ((DenseEmbeddingFloatResults.Embedding) floatResult.chunks().get(0).embedding()).values(),
                        0.0f
                    );
                }
                {
                    assertThat(results.get(1), CoreMatchers.instanceOf(ChunkedInferenceEmbedding.class));
                    var floatResult = (ChunkedInferenceEmbedding) results.get(1);
                    assertThat(floatResult.chunks(), hasSize(1));
                    assertEquals(new ChunkedInference.TextOffset(0, 2), floatResult.chunks().get(0).offset());
                    assertThat(floatResult.chunks().get(0).embedding(), instanceOf(DenseEmbeddingFloatResults.Embedding.class));
                    assertArrayEquals(
                        new float[] { 0.223F, 0.278F },
                        ((DenseEmbeddingFloatResults.Embedding) floatResult.chunks().get(0).embedding()).values(),
                        0.0f
                    );
                }
            }
        }
    }

    private AmazonBedrockService createAmazonBedrockService() {
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
    public InferenceService createInferenceService() {
        return createAmazonBedrockService();
    }

    @Override
    public Model createEmbeddingModel(SimilarityMeasure similarity) {
        return AmazonBedrockEmbeddingsModelTests.createModel(
            randomAlphaOfLength(8),
            randomAlphaOfLength(8),
            randomAlphaOfLength(8),
            AmazonBedrockProvider.AMAZONTITAN,
            null,
            false,
            null,
            similarity,
            null,
            randomAlphaOfLength(8),
            randomAlphaOfLength(8)
        );
    }

    @Override
    public SimilarityMeasure getDefaultSimilarity() {
        return SimilarityMeasure.COSINE;
    }

    private Map<String, Object> getRequestConfigMap(
        Map<String, Object> serviceSettings,
        Map<String, Object> taskSettings,
        Map<String, Object> secretSettings
    ) {
        var builtServiceSettings = new HashMap<>();
        builtServiceSettings.putAll(serviceSettings);
        builtServiceSettings.putAll(secretSettings);

        return new HashMap<>(
            Map.of(ModelConfigurations.SERVICE_SETTINGS, builtServiceSettings, ModelConfigurations.TASK_SETTINGS, taskSettings)
        );
    }
}
