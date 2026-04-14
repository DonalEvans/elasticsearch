/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.deepseek;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.ServiceSettings;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.net.URI;
import java.util.Map;
import java.util.Objects;

import static org.elasticsearch.xpack.inference.services.ServiceFields.MODEL_ID;
import static org.elasticsearch.xpack.inference.services.ServiceFields.URL;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.createOptionalUri;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalString;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractRequiredString;

public record DeepSeekServiceSettings(String modelId, URI uri) implements ServiceSettings {
    public static final String NAME = "deep_seek_service_settings";
    private static final TransportVersion ML_INFERENCE_DEEPSEEK = TransportVersion.fromName("ml_inference_deepseek");

    public DeepSeekServiceSettings {
        Objects.requireNonNull(modelId);
    }

    public DeepSeekServiceSettings(StreamInput in) throws IOException {
        this(in.readString(), in.readOptional(url -> URI.create(url.readString())));
    }

    static DeepSeekServiceSettings fromMap(Map<String, Object> serviceSettingsMap) {
        var validationException = new ValidationException();

        var model = extractRequiredString(serviceSettingsMap, MODEL_ID, ModelConfigurations.SERVICE_SETTINGS, validationException);
        var uri = createOptionalUri(
            extractOptionalString(serviceSettingsMap, URL, ModelConfigurations.SERVICE_SETTINGS, validationException)
        );

        validationException.throwIfValidationErrorsExist();
        return new DeepSeekServiceSettings(model, uri);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        assert false : "should never be called when supportsVersion is used";
        return ML_INFERENCE_DEEPSEEK;
    }

    @Override
    public boolean supportsVersion(TransportVersion version) {
        return version.supports(ML_INFERENCE_DEEPSEEK);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(modelId);
        out.writeOptionalString(uri != null ? uri.toString() : null);
    }

    @Override
    public ToXContentObject getFilteredXContentObject() {
        return this;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(MODEL_ID, modelId);
        if (uri != null) {
            builder.field(URL, uri.toString());
        }
        return builder.endObject();
    }
}
