/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.openai;

import org.elasticsearch.xpack.inference.services.AbstractParsePersistedConfigTests;

public class OpenAiServiceParsePersistedConfigTests extends AbstractParsePersistedConfigTests {
    public OpenAiServiceParsePersistedConfigTests(AbstractParsePersistedConfigTests.TestCase testCase) {
        super(OpenAiServiceParameterizedTestConfiguration.createTestConfiguration(), testCase);
    }
}
