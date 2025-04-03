import js from '@eslint/js';
import tsParser from '@typescript-eslint/parser';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import reactPlugin from 'eslint-plugin-react';
import reactHooksPlugin from 'eslint-plugin-react-hooks';

export default [
    js.configs.recommended,
    {
        files: ['**/*.ts', '**/*.tsx'],
        languageOptions: {
            parser: tsParser,
            parserOptions: {
                ecmaVersion: 2021,
                sourceType: 'module',
                ecmaFeatures: {
                    jsx: true
                },
                project: './tsconfig.json',
                tsconfigRootDir: '.'
            }
        },
        plugins: {
            '@typescript-eslint': tsPlugin,
            'react': reactPlugin,
            'react-hooks': reactHooksPlugin
        },
        rules: {
            // Disable style/formatting rules
            'indent': ['error', 4],
            'linebreak-style': ['error', 'unix'],
            'quotes': 'off',
            'semi': 'off',

            // Focus on important code quality issues
            'no-unused-vars': 'error',
            'no-undef': 'error',
            'no-duplicate-imports': 'error',
            'no-constant-condition': 'error',
            'no-unreachable': 'error',
            'no-unsafe-negation': 'error',
            'no-unsafe-optional-chaining': 'error',
            'no-throw-literal': 'error',

            // TypeScript specific important rules
            '@typescript-eslint/no-explicit-any': 'warn',
            '@typescript-eslint/no-unused-vars': ['error', { 'argsIgnorePattern': '^_' }],
            '@typescript-eslint/no-non-null-assertion': 'warn',
            '@typescript-eslint/no-misused-promises': 'error',
            '@typescript-eslint/no-floating-promises': 'error',
            '@typescript-eslint/no-unnecessary-type-assertion': 'error',

            // React specific important rules
            'react/react-in-jsx-scope': 'off',
            'react/prop-types': 'off',
            'react-hooks/rules-of-hooks': 'error',
            'react-hooks/exhaustive-deps': 'warn'
        },
        settings: {
            react: {
                version: 'detect'
            }
        }
    }
];
