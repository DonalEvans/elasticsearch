/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.amazonbedrock;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.elasticsearch.xpack.inference.services.AbstractBuildModelFromConfigAndSecretsTests;

import java.util.ArrayList;

public class AmazonBedrockServiceBuildModelFromConfigAndSecretsTests extends AbstractBuildModelFromConfigAndSecretsTests {
    public AmazonBedrockServiceBuildModelFromConfigAndSecretsTests(AmazonBedrockProvider provider, TestCase testCase) {
        super(AmazonBedrockServiceParameterizedTestConfiguration.createTestConfiguration(provider), testCase);
    }

    @ParametersFactory
    public static Iterable<Object[]> providersAndTestCases() {
        var providersAndTestCases = new ArrayList<Object[]>();
        for (AmazonBedrockProvider provider : AmazonBedrockProvider.values()) {
            parameters().forEach(testCase -> providersAndTestCases.add(new Object[] { provider, testCase[0] }));
        }
        return providersAndTestCases;
    }
}
