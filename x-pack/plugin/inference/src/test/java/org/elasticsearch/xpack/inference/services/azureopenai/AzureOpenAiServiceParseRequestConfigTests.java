/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.azureopenai;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.elasticsearch.xpack.inference.services.AbstractParseRequestConfigTests;
import org.elasticsearch.xpack.inference.services.azureopenai.AzureOpenAiServiceParameterizedTestConfiguration.AzureOpenAiSecretsTypes;

import java.util.ArrayList;

public class AzureOpenAiServiceParseRequestConfigTests extends AbstractParseRequestConfigTests {
    public AzureOpenAiServiceParseRequestConfigTests(AzureOpenAiSecretsTypes secretsType, TestCase testCase) {
        super(AzureOpenAiServiceParameterizedTestConfiguration.createTestConfiguration(secretsType), testCase);
    }

    @ParametersFactory
    public static Iterable<Object[]> providersAndTestCases() {
        var secretsTypesAndTestCases = new ArrayList<Object[]>();
        for (AzureOpenAiSecretsTypes secretsType : AzureOpenAiSecretsTypes.values()) {
            parameters().forEach(testCase -> secretsTypesAndTestCases.add(new Object[] { secretsType, testCase[0] }));
        }
        return secretsTypesAndTestCases;
    }
}
