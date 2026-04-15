/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services;

/**
 * Base class for testing inference services.
 * <p>
 * This class provides common unit tests for inference services, such as calling the infer method.
 *
 * To use this class, extend it and pass the constructor a configuration.
 * </p>
 */
public abstract class AbstractInferenceServiceTests extends AbstractInferenceServiceBaseTests {

    public AbstractInferenceServiceTests(TestConfiguration testConfiguration) {
        super(testConfiguration);
    }

}
