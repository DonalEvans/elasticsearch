/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.openshiftai;

import org.elasticsearch.xpack.inference.services.AbstractParseRequestConfigTests;

public class OpenShiftAiServiceParseRequestConfigTests extends AbstractParseRequestConfigTests {
    public OpenShiftAiServiceParseRequestConfigTests(TestCase testCase) {
        super(OpenShiftAiServiceParameterizedTestConfiguration.createTestConfiguration(), testCase);
    }
}
