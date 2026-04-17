/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.amazonbedrock;

import org.elasticsearch.xpack.inference.services.AbstractBuildModelFromConfigAndSecretsTests;

public class AmazonBedrockServiceBuildModelFromConfigAndSecretsAi21LabsTests extends AbstractBuildModelFromConfigAndSecretsTests {
    public AmazonBedrockServiceBuildModelFromConfigAndSecretsAi21LabsTests(TestCase testCase) {
        super(AmazonBedrockServiceParameterizedTestConfiguration.createTestConfiguration(AmazonBedrockProvider.AI21LABS), testCase);
    }
}
